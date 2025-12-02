# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, features, labels):
#         """
#         features: [batch_size, projection_dim]
#         labels: [batch_size]
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         batch_size = features.shape[0]
#         labels = labels.contiguous().view(-1, 1)  # shape: (B, 1)

#         mask = torch.eq(labels, labels.T).float().to(device)

#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature
#         )

#         # 防止数值不稳定
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # 去掉自己与自己
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask

#         # 计算 log-softmax
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

#         # 每个 anchor 取其 positive 的 log_prob 平均
#         mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

#         # 损失
#         loss = -mean_log_prob_pos
#         return loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, projection_dim]
        labels: [batch_size]
        """
        device = features.device

        # 添加调试检查
        assert not torch.isnan(features).any(), "features contain NaN"
        assert not torch.isinf(features).any(), "features contain Inf"
        assert features.dim() == 2, f"Expected [B, D], got {features.shape}"
        assert labels.shape[0] == features.shape[0], "Labels and features batch mismatch"

        batch_size = features.shape[0]
        if batch_size == 0:
            raise ValueError("Empty batch passed to SupConLoss")

        features = features.to(device)
        labels = labels.to(device)
        labels = labels.contiguous().view(-1, 1)  # shape: (B, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        diag_indices = torch.arange(batch_size, device=device).view(-1, 1)
        logits_mask.scatter_(1, diag_indices, 0)

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos
        return loss.mean()
