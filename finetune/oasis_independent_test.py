# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                           matthews_corrcoef, cohen_kappa_score, classification_report
import nibabel as nib
import pywt
import torch.nn as nn
import torch.nn.functional as F

# 定义模型类 - 从modelV24.py复制必要的组件
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
        batch_size, channels, _, _, _ = x.size()
        y = self.pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1, 1)
        return x * y.expand_as(x)

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
            nn.Dropout3d(p=dropout),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ECA_Module(nn.Module):
    def __init__(self, channels):
        super(ECA_Module, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        return x

class DeepFusionNetV2(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super().__init__()
        self.low_wavelet = WaveletLowpass()
        self.brain_down = VGGDownBlock(in_channels, out_channels)
        self.hipp_down = VGGDownBlock(in_channels, out_channels)
        self.gated_fusion = GatedFusion(out_channels)
        self.eca = ECA_Module(out_channels)

    def forward(self, b, h):
        b = self.low_wavelet(b)
        h = self.low_wavelet(h)
        b_feat = self.brain_down(b)
        h_feat = self.hipp_down(h)
        fused = self.gated_fusion(b_feat, h_feat)
        fused = self.eca(fused)
        return fused

class VGGStyleEncoder(nn.Module):
    def __init__(self, in_channels=128, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, feature_dim)

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

# 数据集类
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, brain_files, hipp_files, labels, transform=None):
        assert len(brain_files) == len(hipp_files) == len(labels)
        self.samples = list(zip(brain_files, hipp_files, labels))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_path, h_path, label = self.samples[idx]
        b_img = self._normalize(nib.load(b_path).get_fdata()).astype(np.float32)
        h_img = self._normalize(nib.load(h_path).get_fdata()).astype(np.float32)
        b_tensor = torch.from_numpy(b_img).unsqueeze(0)
        h_tensor = torch.from_numpy(h_img).unsqueeze(0)
        if self.transform:
            b_tensor = self.transform(b_tensor)
            h_tensor = self.transform(h_tensor)
        return (b_tensor, h_tensor), label

    def _normalize(self, arr):
        arr = np.nan_to_num(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8) if np.std(arr) != 0 else arr

# 加载OASIS数据
def load_oasis_independent_validation(oasis_brain_root, oasis_hipp_root):
    """加载OASIS独立验证数据集"""
    print("[开始加载OASIS独立验证集]")
    # 收集文件和标签
    brain_paths, hipp_paths, labels = [], [], []
    label_map = {'AD': 0, 'CN': 1, 'MCI': 2}

    for label_name, label_value in label_map.items():
        brain_dir = os.path.join(oasis_brain_root, label_name)
        hipp_dir = os.path.join(oasis_hipp_root, label_name)

        if not os.path.exists(brain_dir):
            print(f"警告：目录不存在: {brain_dir}")
            continue
        if not os.path.exists(hipp_dir):
            print(f"警告：目录不存在: {hipp_dir}")
            continue

        brain_files = sorted([f for f in os.listdir(brain_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

        for b_file in brain_files:
            b_path = os.path.join(brain_dir, b_file)
            # 处理OASIS格式匹配
            if '_regist' in b_file:
                # 移除_processed.nii.gz并添加_L_Hipp.nii.gz
                # 考虑文件名中可能存在的额外点号
                file_base = b_file.split('_regist')[0]
                h_file = f"{file_base}_L_Hipp.nii.gz"
                h_path = os.path.join(hipp_dir, h_file)
            else:
                h_path = None
                h_file = None
            
            if h_path and os.path.exists(h_path):
                brain_paths.append(b_path)
                hipp_paths.append(h_path)
                labels.append(label_value)
            else:
                if h_file:
                    print(f"[警告] 找不到海马图像: {h_file}")

    print(f"[OASIS独立验证集] 加载完成，共 {len(brain_paths)} 个样本")
    
    # 过滤掉增强图像（如果有的话）
    def is_augmented(fname):
        return any(prefix in fname for prefix in ['blur_', 'flip_', 'spike_'])
    
    non_aug_indices = [i for i, f in enumerate(brain_paths) if not is_augmented(os.path.basename(f))]
    brain_paths = [brain_paths[i] for i in non_aug_indices]
    hipp_paths = [hipp_paths[i] for i in non_aug_indices]
    labels = [labels[i] for i in non_aug_indices]
    
    print(f"[OASIS独立验证集] 过滤后，共 {len(brain_paths)} 个样本")
    
    return brain_paths, hipp_paths, labels

# 计算AUC
def compute_auc(labels, probs, dataset_name="验证集"):
    """计算多分类AUC（One-vs-Rest策略）"""
    from sklearn.metrics import roc_auc_score
    try:
        # 确保probs是二维数组
        if len(probs.shape) == 1:
            probs = probs.reshape(-1, 1)
        
        # 检查是否有足够的类别
        num_classes = probs.shape[1]
        if num_classes <= 1:
            print(f"[{dataset_name}] AUC计算失败: 预测概率数组维度不足")
            return np.nan
        
        # 多分类OvR AUC
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        return auc
    except Exception as e:
        print(f"[{dataset_name}] AUC计算失败: {e}")
        return np.nan

# 主测试函数
def test_oasis_with_model(model_path, oasis_brain_root, oasis_hipp_root, result_dir):
    """使用指定模型对OASIS数据集进行独立测试"""
    # 创建结果保存目录
    os.makedirs(result_dir, exist_ok=True)
    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 16,
        'num_classes': 3
    }
    print(f"使用设备: {config['device']}")
    
    # 加载OASIS数据
    oasis_brain_files, oasis_hipp_files, oasis_labels = load_oasis_independent_validation(
        oasis_brain_root, oasis_hipp_root
    )
    
    if len(oasis_brain_files) == 0:
        print("错误：未找到OASIS数据！")
        return
    
    # 创建数据集和数据加载器
    oasis_dataset = EvalDataset(oasis_brain_files, oasis_hipp_files, oasis_labels)
    oasis_loader = DataLoader(
        oasis_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0  # Windows环境下设置为0
    )
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = FullModel().to(config['device'])
    try:
        model.load_state_dict(torch.load(model_path, map_location=config['device']))
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 测试模式
    model.eval()
    
    # 在OASIS数据集上进行预测
    print("在OASIS数据集上进行预测...")
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for (b, h), labels in tqdm(oasis_loader, desc="OASIS预测"):
            b, h, labels = map(lambda x: x.to(config['device']), [b, h, labels])
            
            # 前向传播
            _, logits = model(b, h)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 处理结果数组
    all_probs = np.concatenate(all_probs) if all_probs else np.array([])
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算评估指标
    print("\n" + "="*60)
    print("OASIS独立测试结果")
    print("="*60)
    
    # 准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    
    # 精确率、召回率、F1值（加权平均）
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"加权精确率 (Precision): {precision:.4f}")
    print(f"加权召回率 (Recall): {recall:.4f}")
    print(f"加权F1值 (F1): {f1:.4f}")
    
    # AUC
    auc_score = compute_auc(all_labels, all_probs, "OASIS测试")
    auc_str = f"{auc_score:.4f}" if not np.isnan(auc_score) else "NaN"
    print(f"AUC (OvR, weighted): {auc_str}")
    
    # MCC和Cohen's Kappa
    mcc = matthews_corrcoef(all_labels, all_preds)
    cohen_kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"MCC: {mcc:.4f}")
    print(f"Cohen's Kappa: {cohen_kappa:.4f}")
    
    # 每个类别的指标
    print("\n每个类别的指标:")
    class_names = ["AD", "CN", "MCI"]
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    for i in range(config['num_classes']):
        if i < len(class_names):
            print(f"类别 {class_names[i]}:")
            print(f"  精确率: {precision_per_class[i]:.4f}")
            print(f"  召回率: {recall_per_class[i]:.4f}")
            print(f"  F1值: {f1_per_class[i]:.4f}")
    
    # 生成分类报告
    print("\nOASIS分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 保存结果到文件
    with open(os.path.join(result_dir, "oasis_test_results.txt"), 'w') as f:
        f.write("OASIS独立测试结果\n")
        f.write(f"测试模型: {model_path}\n\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"加权精确率 (Precision): {precision:.4f}\n")
        f.write(f"加权召回率 (Recall): {recall:.4f}\n")
        f.write(f"加权F1值 (F1): {f1:.4f}\n")
        f.write(f"AUC (OvR, weighted): {auc_str}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Cohen's Kappa: {cohen_kappa:.4f}\n\n")
        
        f.write("每个类别的指标:\n")
        for i in range(config['num_classes']):
            if i < len(class_names):
                f.write(f"类别 {class_names[i]}:\n")
                f.write(f"  精确率: {precision_per_class[i]:.4f}\n")
                f.write(f"  召回率: {recall_per_class[i]:.4f}\n")
                f.write(f"  F1值: {f1_per_class[i]:.4f}\n")
        
        f.write("\n分类报告:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print(f"\n结果保存至: {os.path.join(result_dir, 'oasis_test_results.txt')}")
    
    # 尝试导入并使用绘图功能
    try:
        from utils.painter import plot_roc, plot_precision_recall, plot_confusion_matrix
        
        # 绘制ROC曲线
        plot_roc(all_labels, all_probs, config['num_classes'], 0, plots_dir)
        print(f"ROC曲线保存至: {plots_dir}")
        
        # 绘制PR曲线
        plot_precision_recall(all_labels, all_probs, config['num_classes'], 0, plots_dir)
        print(f"PR曲线保存至: {plots_dir}")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(all_labels, all_preds, 0, plots_dir, class_names=class_names)
        print(f"混淆矩阵保存至: {plots_dir}")
        
    except Exception as e:
        print(f"绘图功能导入失败: {e}")
        print("跳过可视化步骤")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'mcc': mcc,
        'cohen_kappa': cohen_kappa,
        'result_dir': result_dir
    }

if __name__ == "__main__":
    # 指定模型路径和数据路径
    MODEL_PATH = r"F:\ADNI\ClassificationAD\PROJECT\CONTRAST_LEARNING-master\runs\experiment_20251024_110747\fold_1\best_model.pth"
    
    # OASIS数据路径
    OASIS_BRAIN_ROOT = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\OAS\OAS_brain\OAS_brain_nyul"
    OASIS_HIPP_ROOT = "F:\\ADNI\\ADNI_PNG_3Ddata\\download_data\\NIFTI_data\\OAS\\OAS_hipp\\Lside_resampled"
    
    # 结果保存路径
    RESULT_DIR = "F:\\ADNI\\ClassificationAD\\PROJECT\\CONTRAST_LEARNING-master\\runs\\oasis_independent_test"
    
    # 运行测试
    test_oasis_with_model(MODEL_PATH, OASIS_BRAIN_ROOT, OASIS_HIPP_ROOT, RESULT_DIR)
    print("\nOASIS独立测试完成！")