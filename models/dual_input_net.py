import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import cv2
from models.resnet18_3d_se import resnet18_3d_se

# ==================================================================================================
# 还没有定义ECA模块和全局-局部融合模块的版本，可以直接运行
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))


# class DeepFusionNet(nn.Module):
#     def __init__(self, in_channels=1, base_channels=32, feature_dim=128):
#         super(DeepFusionNet, self).__init__()

#         # 1x1 conv + bn blocks
#         self.initial_fuse = ConvBlock(in_channels, 1, kernel_size=1, stride=1, padding=0)
#         self.b_conv1 = ConvBlock(in_channels, base_channels, kernel_size=1, stride=1, padding=0)
#         self.h_conv1 = ConvBlock(in_channels, base_channels, kernel_size=1, stride=1, padding=0)

#         #self.b1_conv = ConvBlock(base_channels, base_channels, kernel_size=1, stride=1, padding=0)
#         #self.b2_conv = ConvBlock(base_channels, base_channels, kernel_size=1, stride=1, padding=0)

#         self.c1_conv = ConvBlock(base_channels, base_channels, kernel_size=1, stride=1, padding=0)

#         self.b3_conv = ConvBlock(base_channels, base_channels, kernel_size=1, stride=1, padding=0)
#         self.b4_conv = ConvBlock(base_channels, base_channels, kernel_size=1, stride=1, padding=0)

#         self.c3_conv = ConvBlock(base_channels, base_channels, kernel_size=1, stride=1, padding=0)

#         # Deep feature extraction
#         self.conv7 = ConvBlock(base_channels, base_channels, kernel_size=7, stride=2, padding=3)
#         self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

#         self.res1a = ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
#         self.res1b = ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1)

#         self.res2a = ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
#         self.res2b = ConvBlock(base_channels, base_channels, kernel_size=3, stride=1, padding=1)

#         self.down1 = ConvBlock(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
#         self.down2 = ConvBlock(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.feature_dim = feature_dim
#         self.final_linear = nn.Linear(base_channels, feature_dim)

#     def forward(self, b, h):
#         x = b + h

#         x = self.initial_fuse(x)

#         b1 = self.b_conv1(b + x)
#         #b1 = self.b1_conv(b1)
#         b2 = self.h_conv1(h + x)
#         #b2 = self.b2_conv(b2)

#         c1 = self.c1_conv(b1 + b2)

#         b3 = self.b3_conv(b1 + c1)
#         b4 = self.b4_conv(b2 + c1)

#         c2 = b3 + b4
#         c3 = self.c3_conv(c2)

#         c4 = b + h + c3

#         k1 = self.conv7(c4)
#         k1 = self.pool(k1)

#         x = self.res1a(k1)
#         x = self.res1b(x)
#         k2 = x
#         k3 = k1 + k2

#         x = self.res2a(k3)
#         x = self.res2b(x)
#         k4 = x
#         k5 = k3 + k4

#         x = F.relu(k5)
#         x = self.down1(x)
#         x = self.down2(x)

#         x = self.avgpool(x).flatten(1)
#         x = self.final_linear(x)
#         return x


# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim=128, projection_dim=128):
#         super(ProjectionHead, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.BatchNorm1d(input_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(input_dim, projection_dim)
#         )
#         self.proj = nn.Sequential(
#             nn.Flatten(),  # 变为 [B, C]
#             nn.Linear(input_dim, input_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(input_dim, projection_dim)
#         )

#     def forward(self, x):
#         x = torch.flatten(x, start_dim=1)  # [B, C*D*H*W]
#         x = self.proj(x)  # [B, 128]
#         return self.net(x)
    
# # 分类头
# class ClassificationHead(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=512, num_classes=3):
#         super(ClassificationHead, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, num_classes)  # 注意：不加 softmax
#         )

#     def forward(self, x):
#         return self.classifier(x)


# ==================================================================================================
# 有定义ECA模块和GCF模块的版本，配合的是V4结构
# class ECAModule(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn1 = nn.BatchNorm3d(channels)
#         self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn2 = nn.BatchNorm3d(channels)
#         self.conv3 = nn.Conv3d(channels, channels, kernel_size=1)
#         self.bn3 = nn.BatchNorm3d(channels)

#     def forward(self, x):
#         a2 = self.bn1(self.conv1(x))
#         a3 = x + a2
#         a4 = self.bn2(self.conv2(a3))
#         a5 = a3 + a4
#         a6 = self.bn3(self.conv3(a5))
#         a7 = x + a6
#         return a7


# class DeepFusionNetV2(nn.Module):
#     def __init__(self, in_channels=1, feature_dim=128):
#         super().__init__()
#         self.eca = ECAModule(in_channels)

#         self.conv_b1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.bn_b1 = nn.BatchNorm3d(in_channels)
#         self.conv_b2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.bn_b2 = nn.BatchNorm3d(in_channels)

#         self.eca2 = ECAModule(in_channels)

#         self.conv_b3 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.bn_b3 = nn.BatchNorm3d(in_channels)
#         self.conv_b4 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.bn_b4 = nn.BatchNorm3d(in_channels)

#         self.eca3 = ECAModule(in_channels)

#         self.encoder = resnet18_3d_se(input_channels=in_channels)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(512, feature_dim)

#     def forward(self, b, h):
#         a1 = b + h
#         a7 = self.eca(a1)  # ECA(a1)

#         b1 = self.bn_b1(self.conv_b1(a7 + b))
#         b2 = self.bn_b2(self.conv_b2(a7 + h))

#         c1 = self.eca2(b1 + b2)

#         b3 = self.bn_b3(self.conv_b3(c1 + b1))
#         b4 = self.bn_b4(self.conv_b4(c1 + b2))

#         c3 = self.eca3(b3 + b4)

#         c4 = b + h + c3

#         encoded = self.encoder(c4)  # [B, 512, D, H, W]
#         pooled = self.pool(encoded)  # [B, 512, 1, 1, 1]
#         flat = self.flatten(pooled)  # [B, 512]
#         features = self.fc(flat)     # [B, feature_dim]
#         return features


# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim=128, projection_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, projection_dim),
#             nn.ReLU(),
#             nn.Linear(projection_dim, projection_dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class ClassificationHead(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=512, num_classes=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)

# ==================================================================================================
# 有定义ECA模块和GCF模块的版本，配合的是V5结构
# 局部路径增强（1×1 → 3×3 → 1×1）
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.resnet18_3d_se import resnet18_3d_se
# from models.fusion_blocks import ECA_Module, FusionConvBlock


# class DeepFusionNetV2(nn.Module):
#     def __init__(self, in_channels=1, feature_dim=128):
#         super().__init__()

#         # 初始融合: ECA
#         self.eca1 = ECA_Module(in_channels)

#         # 局部路径融合
#         self.b1_block = FusionConvBlock(in_channels, in_channels)
#         self.b2_block = FusionConvBlock(in_channels, in_channels)

#         # ECA模块 (b1 + b2)
#         self.eca2 = ECA_Module(in_channels)

#         # 局部路径增强
#         self.b3_block = FusionConvBlock(in_channels, in_channels)
#         self.b4_block = FusionConvBlock(in_channels, in_channels)

#         # ECA模块 (b3 + b4)
#         self.eca3 = ECA_Module(in_channels)

#         # ResNet18_3D_SE encoder
#         self.encoder = resnet18_3d_se(input_channels=in_channels)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(512, feature_dim)

#     def forward(self, b, h):
#         a1 = b + h
#         a7 = self.eca1(a1)  # 初始融合

#         # 局部路径融合
#         b1 = self.b1_block(a7 + b)
#         b2 = self.b2_block(a7 + h)

#         # 第一次ECA
#         c1 = self.eca2(b1 + b2)

#         # 局部路径增强
#         b3 = self.b3_block(c1 + b1)
#         b4 = self.b4_block(c1 + b2)

#         # 第二次ECA
#         c3 = self.eca3(b3 + b4)

#         # 总融合
#         c4 = b + h + c3

#         # Encoder
#         encoded = self.encoder(c4)
#         pooled = self.pool(encoded)
#         flat = self.flatten(pooled)
#         features = self.fc(flat)
#         return features


# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim=128, projection_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, projection_dim),
#             nn.ReLU(),
#             nn.Linear(projection_dim, projection_dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class ClassificationHead(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=512, num_classes=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)

# ==================================================================================================

# 注意力模块修改方案
# 在双分支上添加SE，即添加之后变成SE → 1×1Conv → BN → 3×3Conv → BN → 1×1Conv → BN → SE
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.resnet18_3d_se import resnet18_3d_se
# from models.fusion_blocks import ECA_Module, FusionConvBlock


# class DeepFusionNetV2(nn.Module):
#     def __init__(self, in_channels=1, feature_dim=128):
#         super().__init__()

#         # 初始融合: ECA
#         self.eca1 = ECA_Module(in_channels)

#         # 局部路径融合
#         self.b1_block = FusionConvBlock(in_channels, in_channels)
#         self.b2_block = FusionConvBlock(in_channels, in_channels)

#         # ECA模块 (b1 + b2)
#         self.eca2 = ECA_Module(in_channels)

#         # 局部路径增强
#         self.b3_block = FusionConvBlock(in_channels, in_channels)
#         self.b4_block = FusionConvBlock(in_channels, in_channels)

#         # ECA模块 (b3 + b4)
#         self.eca3 = ECA_Module(in_channels)

#         # ResNet18_3D_SE encoder
#         self.encoder = resnet18_3d_se(input_channels=in_channels)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(512, feature_dim)

#     def forward(self, b, h):
#         a1 = b + h
#         a7 = self.eca1(a1)  # 初始融合

#         # 局部路径融合
#         b1 = self.b1_block(a7 + b)
#         b2 = self.b2_block(a7 + h)

#         # 第一次ECA
#         c1 = self.eca2(b1 + b2)

#         # 局部路径增强
#         b3 = self.b3_block(c1 + b1)
#         b4 = self.b4_block(c1 + b2)

#         # 第二次ECA
#         c3 = self.eca3(b3 + b4)

#         # 总融合
#         c4 = b + h + c3

#         # Encoder
#         encoded = self.encoder(c4)
#         pooled = self.pool(encoded)
#         flat = self.flatten(pooled)
#         features = self.fc(flat)
#         return features


# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim=128, projection_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, projection_dim),
#             nn.ReLU(),
#             nn.Linear(projection_dim, projection_dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class ClassificationHead(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=512, num_classes=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)


# 在ECA模块之前将逐元素相加改成，非零元素相乘，零元素相加
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_3d_se import resnet18_3d_se
from models.fusion_blocks import ECA_Module, FusionConvBlock

def conditional_add_or_mul(b, h):
    # b, h are tensors of shape [B, C, D, H, W]
    mask_b = (b != 0).float()
    mask_h = (h != 0).float()
    both_nonzero = (mask_b * mask_h)  # 两个都不为0的位置 = 1
    not_both_nonzero = 1 - both_nonzero  # 其他位置 = 1

    mul_part = b * h * both_nonzero  # 两者非零位置乘法
    add_part = (b + h) * not_both_nonzero  # 其他位置加法
    return mul_part + add_part


class DeepFusionNetV2(nn.Module):
    def __init__(self, in_channels=1, feature_dim=128):
        super().__init__()

        # 初始融合: ECA
        self.eca1 = ECA_Module(in_channels)

        # 局部路径融合
        self.b1_block = FusionConvBlock(in_channels, in_channels)
        self.b2_block = FusionConvBlock(in_channels, in_channels)

        # ECA模块 (b1 + b2)
        self.eca2 = ECA_Module(in_channels)

        # 局部路径增强
        self.b3_block = FusionConvBlock(in_channels, in_channels)
        self.b4_block = FusionConvBlock(in_channels, in_channels)

        # ECA模块 (b3 + b4)
        self.eca3 = ECA_Module(in_channels)

        # ResNet18_3D_SE encoder
        self.encoder = resnet18_3d_se(input_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, b, h):

        a1 = conditional_add_or_mul(b, h)  # 替换掉原来的 a1 = b + h
        a7 = self.eca1(a1)

        # 局部路径融合
        b1 = self.b1_block(a7 + b)
        b2 = self.b2_block(a7 + h)

        # 第一次ECA
        x = conditional_add_or_mul(b1, b2)
        c1 = self.eca2(x)

        # 局部路径增强
        b3 = self.b3_block(c1 + b1)
        b4 = self.b4_block(c1 + b2)

        # 第二次ECA
        x = conditional_add_or_mul(b3, b4)
        c3 = self.eca3(x)

        # 总融合
        c4 = b + h + c3

        # Encoder
        encoded = self.encoder(c4)
        pooled = self.pool(encoded)
        flat = self.flatten(pooled)
        features = self.fc(flat)
        return features


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)