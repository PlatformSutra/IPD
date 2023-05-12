import torch
import torch.nn as nn
import cv2
from IPBNet_ALL.IPBNet.model.deconv import FastDeconv


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        #print(fea1.shape,fea2.shape)
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class Dehaze(nn.Module):
    def __init__(self):
        super(Dehaze, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(3, 64, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block1 = DehazeBlock(default_conv, 256, 3)
        self.block2 = DehazeBlock(default_conv, 256, 3)
        self.block3 = DehazeBlock(default_conv, 256, 3)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(64, 3, kernel_size=7, padding=0),
                                 nn.Tanh())

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, input):
        x_deconv = self.deconv(input)
        x_down1 = self.down1(x_deconv)
        #print(x_deconv.shape)
        x_down2 = self.down2(x_down1)
        #print(x_down2.shape)
        x_down3 = self.down3(x_down2)
        x1 = self.block1(x_down3)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        #print(x_down3.shape,x2.shape)
        x_out_mix = self.mix1(x_down3, x3)
        #print(x_out_mix.shape)
        x_up1 = self.up1(x_out_mix)
        #print(x_up1.shape,x_down2.shape)
        x_up1_mix = self.mix2(x_down2, x_up1)
        #print(x_up1_mix.shape)
        x_up2 = self.up2(x_up1_mix)
        out = self.up3(x_up2)
        return out


class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(6, 60, 1, padding=0, bias=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(60, 3, 7, padding=0, bias=True),
            nn.Tanh()
        )
    def forward(self, input):
        out = self.fusion(input)
        return out


class HSV(nn.Module):
    def __init__(self):
        super(HSV, self).__init__()
    def forward(self, input):
        input = input.permute(0, 2, 3, 1).cpu().detach().numpy()
        for i,it in enumerate(input):
            input[i] = cv2.cvtColor(it, cv2.COLOR_BGR2HSV)
        v = input[:,:, :, 2]
        s = input[:,:, :, 1]
        out_v = v.reshape(v.shape[0],v.shape[1],v.shape[2],1)
        out_v = torch.from_numpy(out_v.transpose((0, 3, 1, 2)))
        out_v = out_v.cuda()
        out_s = v.reshape(s.shape[0], s.shape[1], s.shape[2], 1)
        out_s = torch.from_numpy(out_s.transpose((0, 3, 1, 2)))
        out_s = out_s.cuda()
        out = torch.cat([out_v, out_s], 1)
        return out

class ASM(nn.Module):
    def __init__(self):
        super(ASM, self).__init__()
    def forward(self, input1, input2):
        out = input1 * input2 - input2 + 1
        return out


class IPDNet(nn.Module):
    def __init__(self, channel):
        super(IPDNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, bias=True, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.HSV = HSV()
        self.ASM = ASM()
        self.Dehaze = Dehaze()
        self.fusion = fusion()

    def forward(self, x):
        x1 = self.HSV(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x1 = self.conv7(x1)
        x1 = self.conv8(x1)
        x1 = self.conv9(x1)
        x1 = self.conv10(x1)
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)
        x1 = self.conv14(x1)
        x1 = self.conv15(x1)
        x1 = self.conv16(x1)
        x1 = self.conv17(x1)
        x1 = self.conv18(x1)
        x1 = self.ASM(x, x1)
        x2 = self.Dehaze(x)
        x3 = torch.cat([x1, x2], 1)
        x3 = self.fusion(x3)
        out = x3
        return x1, x2, out
