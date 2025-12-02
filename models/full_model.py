#====================================================================
# V2模型采用的版本

# import torch
# import torch.nn as nn
# from models.dual_input_net import DeepFusionNet, ProjectionHead, ClassificationHead
# from models.resnet18_3d_se import resnet18_3d_se


# class FullModel(nn.Module):
#     def __init__(self):
#         super(FullModel, self).__init__()
#         self.backbone = DeepFusionNet(in_channels=1, base_channels=32, feature_dim=128)
#         self.proj_head = ProjectionHead(input_dim=128, projection_dim=128)
#         self.cls_head = ClassificationHead(input_dim=128, hidden_dim=512, num_classes=3)

#     def forward(self, b, h):
#         features = self.backbone(b, h)       # [B, 128]
#         projections = self.proj_head(features)  # [B, 128]
#         logits = self.cls_head(features)     # [B, 3]
#         return projections, logits
#====================================================================


#====================================================================
# V3模型采用的版本
# import torch
# import torch.nn as nn
# from models.dual_input_net import DualInputFusion, ProjectionHead, ClassificationHead
# from models.resnet18_3d_se import resnet18_3d_se


# class FullModel(nn.Module):
#     def __init__(self):
#         super(FullModel, self).__init__()
#         # 融合模块（你原先的局部全局融合模块）
#         self.fusion = DualInputFusion(in_channels=1)  # 需要你在 dual_input_net.py 中定义

#         # 替换原先的 backbone：使用 ResNet3D-SE
#         self.encoder = resnet18_3d_se(input_channels=1)

#         # ProjectionHead 和 ClassificationHead
#         self.proj_head = ProjectionHead(input_dim=512, projection_dim=128)
#         self.cls_head = ClassificationHead(input_dim=512, hidden_dim=512, num_classes=3)

#     def forward(self, b, h):
#         c4 = self.fusion(b, h)               # 融合输出 [B, 1, D, H, W]
#         features = self.encoder(c4)         # 编码器输出 [B, 512, d', h', w']
#         features = nn.AdaptiveAvgPool3d(1)(features).view(features.size(0), -1)  # 展平

#         projections = self.proj_head(features)  # 对比学习投影 [B, 128]
#         logits = self.cls_head(features)        # 分类输出 [B, 3]
#         return projections, logits


#====================================================================
#V4,V5模型采用的版本
import torch
import torch.nn as nn
from models.dual_input_net import DeepFusionNetV2, ProjectionHead, ClassificationHead



class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DeepFusionNetV2(in_channels=1, feature_dim=128)
        self.proj_head = ProjectionHead(input_dim=128, projection_dim=128)
        self.cls_head = ClassificationHead(input_dim=128, hidden_dim=512, num_classes=3)

    def forward(self, b, h):
        features = self.backbone(b, h)
        projections = self.proj_head(features)
        logits = self.cls_head(features)
        return projections, logits
