# import torch
# import torch.nn as nn

# class ECA_Module(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn1 = nn.BatchNorm3d(channels)
#         self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn2 = nn.BatchNorm3d(channels)
#         self.conv3 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn3 = nn.BatchNorm3d(channels)

#     def forward(self, x):
#         a1 = x
#         a2 = self.bn1(self.conv1(a1))
#         a3 = a1 + a2
#         a4 = self.bn2(self.conv2(a3))
#         a5 = a3 + a4
#         a6 = self.bn3(self.conv3(a5))
#         a7 = a1 + a6
#         return a7

# # 原来是 1×1Conv + BN，改成：1×1Conv + BN → 3×3Conv + BN → 1×1Conv + BN
# class FusionConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
#         self.bn3 = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.relu(self.bn3(self.conv3(out)))
#         return out


# # models/dual_input_net.py 中 DeepFusionNet 修改示意：
# # - 添加 import
# # from models.fusion_blocks import ECA_Module, FusionConvBlock
# # - 替换原融合结构为上述模块的组合

# # main.py 中保持原样，不需要修改


# # ============================================================================================
# 方案3：方案3其实不止在每个ECA之后添加了注意力模块，ECA模块在定义的时候，最开始也会先经过一个SE模块
# import torch
# import torch.nn as nn
# from models.attention import SEBlock
# #from models.attention import CBAMBlock

# class ECAModule(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.att = SEBlock(channels)  # 替换成 SE 注意力模块

#         self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn1 = nn.BatchNorm3d(channels)
#         self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn2 = nn.BatchNorm3d(channels)
#         self.conv3 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn3 = nn.BatchNorm3d(channels)

#     def forward(self, x):
#         x = self.att(x)  # 注意力模块放在最开始
#         a2 = self.bn1(self.conv1(x))
#         a3 = x + a2
#         a4 = self.bn2(self.conv2(a3))
#         a5 = a3 + a4
#         a6 = self.bn3(self.conv3(a5))
#         a7 = x + a6
#         return a7

# # 1×1Conv + BN → 3×3Conv + BN → 1×1Conv + BN
# class FusionConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
#         self.bn3 = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.relu(self.bn3(self.conv3(out)))
#         return out

# ===========================================================================
#方案4的想法是，在双分支上添加SE，即添加之后变成SE → 1×1Conv → BN → 3×3Conv → BN → 1×1Conv → BN → SE
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SEBlock

class ECA_Module(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(channels)

    def forward(self, x):
        a1 = x
        a2 = self.bn1(self.conv1(a1))
        a3 = a1 + a2
        a4 = self.bn2(self.conv2(a3))
        a5 = a3 + a4
        a6 = self.bn3(self.conv3(a5))
        a7 = a1 + a6
        return a7

class FusionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionConvBlock, self).__init__()
        self.se1 = SEBlock(in_channels)  # 前置SE注意力
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.se2 = SEBlock(out_channels)  # 后置SE注意力
        self.dropout = nn.Dropout3d(p=0.2)  # ⭐ 添加 dropout

    def forward(self, x):
        x = self.se1(x)                      # 1st SE
        x = self.bn1(self.conv1(x))         # 1x1 Conv + BN
        x = self.bn2(self.conv2(x))         # 3x3 Conv + BN
        x = self.bn3(self.conv3(x))         # 1x1 Conv + BN
        x = self.se2(x)                      # 2nd SE
        x = self.dropout(x)  # ⭐ dropout 放在最后
        return x
