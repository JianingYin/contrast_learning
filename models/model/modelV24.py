import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# 示例：在DeepFusionNetV2内部添加 DWT 低频通道（PyWavelets方式）
import pywt
import torch.nn.functional as F

class WaveletLowpass(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        out = []
        for i in range(B):
            # 只取低频LLL分量
            coeffs = pywt.dwtn(x[i, 0].cpu().numpy(), self.wave, axes=(0, 1, 2))
            lll = coeffs['aaa']
            lll_tensor = torch.tensor(lll, dtype=x.dtype, device=x.device).unsqueeze(0)
            out.append(lll_tensor)
        return torch.stack(out)  # [B, 1, D//2, H//2, W//2]



class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.se = SEBlock(channels)

    def forward(self, x1, x2):
        fusion_input = torch.cat([x1, x2], dim=1)   # [B, 2C, D, H, W]
        gate = self.gate_conv(fusion_input)         # [B, C, D, H, W]
        out = gate * x1 + (1 - gate) * x2           # 动态加权融合
        out = self.se(out)
        return out
class VGGDownBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),  # ✅ 添加 Dropout
            nn.MaxPool3d(2),  # → [B, 32, 45, 54, 45]

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # → [B, 64, 22, 27, 22]

            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)  # 输出维度: [B, 128, 22, 27, 22]


class DeepFusionNetV2(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super().__init__()
        self.low_wavelet = WaveletLowpass() # 添加小波变换

        self.brain_down = VGGDownBlock(in_channels, out_channels)
        self.hipp_down = VGGDownBlock(in_channels, out_channels)
        self.gated_fusion = GatedFusion(out_channels)     # 通道数为 128
        self.eca = ECA_Module(out_channels)

    def forward(self, b, h):
        b = self.low_wavelet(b)   # 数据传入后先进行一次小波变换
        h = self.low_wavelet(h)   # 数据传入后先进行一次小波变换

        b_feat = self.brain_down(b)   # [B, 128, 22, 27, 22]
        h_feat = self.hipp_down(h)   # [B, 128, 22, 27, 22]
        fused = self.gated_fusion(b_feat, h_feat)  # [B, 128, 22, 27, 22]
        fused = self.eca(fused)
        return fused


class VGGStyleEncoder(nn.Module):
    def __init__(self, in_channels=128, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=3, stride=1, padding=1),  # → [B,128,22,27,22]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),          # → [B,128,11,14,11]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),          # → [B,128,6,7,6]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),          # → [B,128,3,4,3]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(1),                                          # → [B,128,1,1,1]
        )
        self.flatten = nn.Flatten()                                           # → [B,128]
        self.fc = nn.Linear(128, feature_dim)                                # Optional projection to fixed dim

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x




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



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _, _ = x.size()  # ✅ 获取 b 和 c
        y = self.pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1, 1)
        return x * y.expand_as(x)

# 方案二：使用单层卷积 + 投影层（极简）

# class TinyEncoder(nn.Module):
#     def __init__(self, in_channels=1, feature_dim=128):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool3d(1),
#         )
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(32, feature_dim)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x
# 方案一：简化版卷积特征提取器
# class TinyEncoder(nn.Module):
#     def __init__(self, in_channels=1, feature_dim=128):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(16)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm3d(32)

#         self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm3d(64)

#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(64, feature_dim)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x


class ECA_Module(nn.Module):
    def __init__(self, channels):
        super(ECA_Module, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(channels)
        #self.se = SEBlock(channels)  # 后置SE注意力
        #self.dropout = nn.Dropout3d(p=0.3)  # 添加dropout

    def forward(self, x):
        x = self.bn1(self.conv1(x))  # 1×1 Conv + BN
        x = self.bn2(self.conv2(x))  # 3×3 Conv + BN
        x = self.bn3(self.conv3(x))  # 1×1 Conv + BN
        #x = self.se(x)               # SE模块
        #x = self.dropout(x)          # Dropout
        return x


class FusionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionConvBlock, self).__init__()
        #self.se1 = SEBlock(in_channels)  # 前置SE注意力
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.se2 = SEBlock(out_channels)  # 后置SE注意力
        self.dropout = nn.Dropout3d(p=0.3)  # ⭐ 添加 dropout

    def forward(self, x):
        #x = self.se1(x)                      # 1st SE
        x = self.bn1(self.conv1(x))         # 1x1 Conv + BN
        x = self.bn2(self.conv2(x))         # 3x3 Conv + BN
        x = self.bn3(self.conv3(x))         # 1x1 Conv + BN
        x = self.se2(x)                      # 2nd SE
        x = self.dropout(x)                 # ⭐ 添加 dropout
        return x


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DeepFusionNetV2(in_channels=1, out_channels=128)
        self.encoder = VGGStyleEncoder(in_channels=128, feature_dim=128)

        self.proj_head = ProjectionHead(input_dim=128)
        self.cls_head = ClassificationHead(input_dim=128, hidden_dim=512, num_classes=3)


    def forward(self, b, h):
        mid_feat = self.backbone(b, h)
        features = self.encoder(mid_feat)
        projections = self.proj_head(features)
        logits = self.cls_head(features)
        return projections, logits



if __name__ == '__main__':
    # 实例化模型
    model = FullModel()
    exp_input1 = torch.randn(3,1,91,109,91)
    exp_input2 = torch.randn(3,1,91,109,91)
    output,logits = model(exp_input1, exp_input2)
    print(output.shape)
    print(logits.shape)