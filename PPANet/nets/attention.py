import torch
import torch.nn as nn
import math
from collections import OrderedDict

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]#列表推导式
        features = torch.cat(features + [x], dim=1)

        return features

#----------------------------------------------------------------------------------------------------------

class mca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2,pool_sizes=[5,9]):
        super(mca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1


        self.pools_1 = nn.MaxPool2d(pool_sizes[0], 1, pool_sizes[0] // 2)
        self.pools_2 = nn.MaxPool2d(pool_sizes[1], 1, pool_sizes[1] // 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y_0 = self.avg_pool(x)
        y_0 = self.conv(y_0.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_1 =  self.pools_1(x)
        y_1 = self.avg_pool(x_1)
        y_1 = self.conv(y_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) #得到3*3池化的权重
        x_2 =  self.pools_2(x)
        y_2 = self.avg_pool(x_2)
        y_2 = self.conv(y_2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) #得到5*5池化的权重
        y   = y_0 + y_1 + y_2
        y = self.sigmoid(y)
        return x * y.expand_as(x)
#----------------------------------------------------------------------------------------------------------

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 利用1x1卷积代替全连接
#         self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
# class cbam_block(nn.Module):
#     def __init__(self, channel, ratio=8, kernel_size=7):
#         super(cbam_block, self).__init__()
#         self.channelattention = ChannelAttention(channel, ratio=ratio)
#         self.spatialattention = SpatialAttention(kernel_size=kernel_size)
#
#     def forward(self, x):
#         x = x * self.channelattention(x)
#         x = x * self.spatialattention(x)
#         return x


