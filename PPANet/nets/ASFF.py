import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

class ASFF(nn.Module):
    def __init__(self, in_dim=512,rfb=False, vis=False, act_cfg=True):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super(ASFF, self).__init__()
        self.in_dim = in_dim
        #self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')



        # 添加 rfb 时，我们使用一半的通道数来节省内存
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.in_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.in_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 2, 2, 1, 1)
        self.expand = Conv(self.in_dim, self.in_dim
            , 3, 1)
        self.vis = vis

    def forward(self, x0,x1,):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x0  # 最小特征层
        x_level_1 = x1  # 中间特征层


        #x_level_0 = self.Upsample(x_level_0)


        level_0_weight_v = self.weight_level_0(x_level_0)
        level_1_weight_v = self.weight_level_1(x_level_1)


        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)



        fused_out_reduced = x_level_0 * levels_weight[:, 0:1, :, :] + \
                            x_level_1 * levels_weight[:, 1:2, :, :]
        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

