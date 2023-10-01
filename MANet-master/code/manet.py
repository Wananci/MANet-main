from MobileNetV3 import *
import torch
import torch.nn as nn
from convblock import RFB
import torch.nn.functional as F
import math
import torch.fft as fft
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True, swish=False):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        if swish == True and relu == False:
            conv.append(nn.Hardswish())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DWConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DWConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class DWConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DWConv5x5, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=5, s=stride, p=2 * dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)

class ExpandBlock(nn.Module):
    def __init__(self, inp, hidden_dim):
        super(ExpandBlock, self).__init__()
        # hidden_dim = int(round(inp * expand_ratio))
        # layers = []
        # if expand_ratio != 1:
        #     # pw
        #     layers.append(convbnrelu(inp, hidden_dim, k=1, p=0))
        # layers.extend([
        #     # dw
        #     convbnrelu(hidden_dim, hidden_dim, s=1, g=hidden_dim),
        #     # pw-linear
        #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(oup),
        # ])
        # self.conv = nn.Sequential(*layers)
        self.conv = nn.Sequential(convbnrelu(inp, hidden_dim, k=1, p=0),
                                  convbnrelu(hidden_dim, hidden_dim, s=1, g=hidden_dim))
    def forward(self, x):
        return self.conv(x)

class ReductionBlock(nn.Module):
    def __init__(self, hidden_dim, oup):
        super(ReductionBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(oup))
    def forward(self, x):
        return self.conv(x)


class InvertedChannelAttention(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedChannelAttention, self).__init__()
        self.num_expand = len(expand_ratio)
        self.conv = convbnrelu(inp, inp, 1, 1, 0, relu=False)
        self.expand_ivb = nn.ModuleList([])
        self.reduce_ivb = nn.ModuleList([])
        for i in range(0, self.num_expand - 1):
            self.expand_ivb.append(ExpandBlock(inp * expand_ratio[i], inp * expand_ratio[i + 1]))
            self.reduce_ivb.append(ReductionBlock(inp * expand_ratio[i + 1], oup))

    def forward(self, x):
        x = self.conv(x)
        out = x.clone()
        for i in range(0, self.num_expand - 1):
            x = self.expand_ivb[i](x)
            out = torch.cat([out, self.reduce_ivb[i](x)], dim=1)
        return out

class MultiresolutionSpatialAttention(nn.Module):
    def __init__(self, inc, reduce_ratio):
        super(MultiresolutionSpatialAttention, self).__init__()
        self.num_reduce = len(reduce_ratio)
        self.dwconv = nn.ModuleList([])
        for i in range(0, self.num_reduce - 1):
            if i != self.num_reduce - 2:
                self.dwconv.append(DWConv3x3(inc, inc, stride=2))
            else:
                self.dwconv.append(DWConv3x3(inc, inc, stride=2, relu=False))
        self.conv = DWConv3x3(inc, inc)

    def forward(self, x):
        x = self.conv(x)
        out = x
        for i in range(0, self.num_reduce - 1):
            x = F.interpolate(self.dwconv[i](x), size=x.size()[2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, x], dim=1)
        return out

class HybridAttentionLayer(nn.Module):
    def __init__(self, inp, oup, expend_ratio, reduce_ratio):
        super(HybridAttentionLayer, self).__init__()
        # channel_attn
        self.c = InvertedChannelAttention(inp=inp, oup=oup, expand_ratio=expend_ratio)
        self.conv1x1_c1 = nn.Conv2d(len(expend_ratio) * inp, inp, 1)
        self.fc1 = nn.Conv2d(inp, oup, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # spatial_attn
        self.conv1x1_s1 = nn.Conv2d(len(reduce_ratio) * inp, inp, 1)
        self.mlp = nn.Sequential(nn.Conv2d(inp, inp // 16, kernel_size=1, padding=0, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(inp // 16, inp, kernel_size=1, padding=0, bias=False))
        self.conv1x1_s2 = nn.Sequential(convbnrelu(inp, inp // 8),
                                        DWConv3x3(inp // 8, inp // 8, dilation=2),
                                        nn.Conv2d(inp // 8, 1, 1, bias=False))
        self.s = MultiresolutionSpatialAttention(inc=inp, reduce_ratio=reduce_ratio)

    def forward(self, x):
        channel_attn1 = self.c(x)
        channel_attn1 = self.conv1x1_c1(channel_attn1)
        channel_attn1 = self.fc1(self.gap(channel_attn1))  # B C 1 1

        spatial_attn1 = self.s(x)
        spatial_attn1 = self.mlp(self.conv1x1_s1(spatial_attn1))
        spatial_attn1 = self.conv1x1_s2(spatial_attn1)  # B 1 H H

        # union attn
        channel_attn1 = channel_attn1.unsqueeze(1)
        spatial_attn1 = spatial_attn1.unsqueeze(1)
        union_attn = (channel_attn1*spatial_attn1).squeeze(1)
        union_attn = F.softmax(union_attn, dim=1)

        return union_attn * x + x

class HybridAttionNetwork(nn.Module):
    def __init__(self, flag):
        super(HybridAttionNetwork, self).__init__()
        self.mobilenetv3 = MobileNetV3_Large()  # 96 96 24  48 48 40  24 24 80  12 12 160
        self.layer1 = nn.Sequential(
            HybridAttentionLayer(inp=24, oup=24, expend_ratio=[1, 2, 3, 6], reduce_ratio=[8, 4, 2, 1])
        )
        self.layer2 = nn.Sequential(
            HybridAttentionLayer(inp=40, oup=40, expend_ratio=[1, 2, 3, 6], reduce_ratio=[4, 2, 1]),
            HybridAttentionLayer(inp=40, oup=40, expend_ratio=[1, 2, 3, 6], reduce_ratio=[4, 2, 1])
        )
        self.layer3 = nn.Sequential(
            HybridAttentionLayer(inp=80, oup=80, expend_ratio=[1, 2, 3], reduce_ratio=[2, 1]),
            HybridAttentionLayer(inp=80, oup=80, expend_ratio=[1, 2, 3], reduce_ratio=[2, 1]),
            HybridAttentionLayer(inp=80, oup=80, expend_ratio=[1, 2, 3], reduce_ratio=[2, 1])
        )
        self.layer4 = nn.Sequential(
            HybridAttentionLayer(inp=160, oup=160, expend_ratio=[1, 2], reduce_ratio=[2, 1])
        )
        self.fuse = nn.ModuleList([
            DWConv3x3(160, 80, dilation=2),
            DWConv3x3(80, 40, dilation=2),
            DWConv3x3(40, 24, dilation=2),
        ])
        self.out = nn.ModuleList([
            nn.Conv2d(24, 1, 1, 1, 0),
            nn.Conv2d(40, 1, 1, 1, 0),
            nn.Conv2d(80, 1, 1, 1, 0),
            nn.Conv2d(160, 1, 1, 1, 0)
        ])
        self.flag = flag
    def forward(self, x):
        x1, x2, x3, x4 = self.mobilenetv3(x)
        x4 = self.layer4(x4)
        x4_3 = F.interpolate(self.fuse[0](x4), size=x3.size()[2:], mode='bilinear', align_corners=True)

        x3 = self.layer3(x3 + x4_3)
        x3_2 = F.interpolate(self.fuse[1](x3), size=x2.size()[2:], mode='bilinear', align_corners=True)

        x2 = self.layer2(x2 + x3_2)
        x2_1 = F.interpolate(self.fuse[2](x2), size=x1.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.layer1(x1 + x2_1)

        out = F.interpolate(self.out[0](x1), size=x.size()[2:], mode='bilinear', align_corners=True)
        out21 = F.interpolate(self.out[1](x2), size=x.size()[2:], mode='bilinear', align_corners=True)
        out32 = F.interpolate(self.out[2](x3), size=x.size()[2:], mode='bilinear', align_corners=True)
        out43 = F.interpolate(self.out[3](x4), size=x.size()[2:], mode='bilinear', align_corners=True)

        # edge loss
        if self.flag == 'train':
            img1 = fft.fft2(self.out[0](x1), dim=(-2, -1))
            img1_shift = fft.fftshift(img1)
            mask1 = self.mask_radial(self.out[0](x1)).cuda()
            img_shi1_mask = img1_shift * (mask1)
            img_fft1 = fft.ifftshift(img_shi1_mask)
            edge = fft.ifft2(img_fft1, dim=(-2, -1))
            edge = -torch.abs(edge)
            edge = F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=True)

            return out, out21, out32, out43, edge
        else:
            return out, out21, out32, out43


    def mask_radial(self, img):
        batch, channels, wid, hei = img.shape
        mask = torch.zeros((wid, hei), dtype=torch.float32)
        for i in range(wid):
            for j in range(hei):
                mask[i, j] = self.distance(i, j, wid, hei)
        return mask

    def distance(self, i, j, wid, hei):
        dis = math.exp((-(i - wid / 2) ** 2 - (j - hei / 2) ** 2) / 100)
        return dis

    def load_pre(self, pre_model):
        self.mobilenetv3.load_state_dict(torch.load(pre_model), strict=False)
        print(f"loading pre_model ${pre_model}")

