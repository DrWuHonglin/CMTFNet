import torchvision
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

__all__ = ["ResNet18", "ResNet50", "ResNet34"]

#定义一个3*3的卷积模板，步长为1，并且使用大小为1的zeropadding
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
#定义基础模块BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.downsamples = nn.Sequential(
            nn.Conv2d(inplanes,planes,1,1,bias=False),
            nn.BatchNorm2d(planes)
        )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsamples(x)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention=4
    def __init__(self,inplanes,planes):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=1, bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv_sfle = Conv_Self(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        self.downsamples = nn.Sequential(
            nn.Conv2d(inplanes, planes*self.extention, 1, 1, bias=False),
            nn.BatchNorm2d(planes*self.extention)
        )

    def forward(self,x):
        #参差数据
        residual = self.downsamples(x)

        #卷积操作
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out = self.conv_sfle(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)


        #将残差部分和卷积部分相加
        out+=residual
        out=self.relu(out)

        return out

class Conv_Self(nn.Module):
    def __init__(self, in_dim):
        super(Conv_Self, self).__init__()
        self.qkv_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=3, stride=1, padding=1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Conv2d(in_dim // 2, in_dim, kernel_size=1)

    def forward(self, x):
        B, C, W, H = x.shape
        short = x
        q = self.qkv_conv(x)
        k = self.qkv_conv(x)
        v = self.qkv_conv(x)

        v_g = self.GAP(v)
        v_s = self.sigmoid(v_g)
        v = v_s * v
        q = q.view(B, -1, W * H).permute(0, 2, 1)  # B X (HW) X C/2
        k = k.view(B, -1, W * H)  # B X C/2 X (HW)
        energy = torch.bmm(q, k)  # transpose check B X (HW) X (HW)
        attention = self.softmax(energy)  #B X (HW) X (HW)
        v = v.view(B, -1, W * H)  # B X C/2 X (HW)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, -1, W, H)
        out = self.conv(out)
        out = short + out

        return out

class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4



class ResNet34(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        pretrained = torchvision.models.resnet34(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


# flops:21.47G params:23.51M
class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x))) #[1, 64, 256, 256]
        b = self.maxpool(b0) #[1, 64, 128, 128]
        b1 = self.layer1(b) #[1, 256, 128, 128]
        b2 = self.layer2(b1) #[1, 512, 64, 64]
        b3 = self.layer3(b2) #[1, 1024, 32, 32]
        b4 = self.layer4(b3) #[1, 2048, 16, 16]

        return b1, b2, b3, b4



class resnext50_32x4d(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnext50_32x4d, self).__init__()
        pretrained = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool


class resnet152(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnet152, self).__init__()
        pretrained = torchvision.models.resnet152(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool

if __name__ == "__main__":
    from thop import profile
    x = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    net = ResNet50()
    print(net)
    out = net(x)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
