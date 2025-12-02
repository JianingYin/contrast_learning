# -*- coding: utf-8 -*-

# 基础导入
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 导入train_test_split用于分层采样

# 设置随机种子以确保实验可重复性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 数据处理
import nibabel as nib
import pywt

# 评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                           matthews_corrcoef, cohen_kappa_score, classification_report, \
                           confusion_matrix, roc_auc_score

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
            nn.Dropout(p=0.3),
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

# 早停机制类
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# OASIS微调函数 - 仅使用小比例样本微调分类器
def finetune_oasis_model(pretrained_model_path, oasis_brain_root, oasis_hipp_root, output_dir):
    """
    在OASIS数据集上微调预训练模型 - 仅使用小比例样本微调分类器
    目标：让模型的BatchNorm/最后几层参数对齐OASIS分布
    
    Args:
        pretrained_model_path: 预训练模型路径
        oasis_brain_root: OASIS脑图像根目录
        oasis_hipp_root: OASIS海马图像根目录
        output_dir: 输出目录
    
    Returns:
        dict: 包含微调结果的字典
    """
    print("="*60)
    print("开始OASIS小比例样本微调流程")
    print("目标: 仅训练分类器，冻结encoder")
    print("="*60)
    
    # 检查路径是否存在
    if not os.path.exists(pretrained_model_path):
        print(f"错误：预训练模型文件不存在: {pretrained_model_path}")
        return None
    
    if not os.path.exists(oasis_brain_root):
        print(f"错误：OASIS脑图像目录不存在: {oasis_brain_root}")
        return None
    
    if not os.path.exists(oasis_hipp_root):
        print(f"错误：OASIS海马图像目录不存在: {oasis_hipp_root}")
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "finetuned_model.pth")
    
    # 配置 - 使用更小的学习率
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 4,  # 小样本使用更小的批次
        'num_classes': 3,
        'num_epochs': 50,
        'learning_rate': 1e-4,  # 提高学习率到1e-4以便更好地收敛
        'weight_decay': 1e-3,  # 添加权重衰减
        'patience': 50,  # 早停耐心值，设置为10个epocha
        'fine_tune_percentage': 0.15  # 使用总样本的15%进行微调
    }
    print(f"使用设备: {config['device']}")
    print(f"配置: {config}")
    
    # 加载OASIS数据
    print(f"\n加载OASIS数据...")
    try:
        oasis_brain_files, oasis_hipp_files, oasis_labels = load_oasis_independent_validation(
            oasis_brain_root, oasis_hipp_root
        )
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None
    
    if len(oasis_brain_files) == 0:
        print("错误：未找到OASIS数据！")
        return None
    
    print(f"成功加载 {len(oasis_brain_files)} 个OASIS样本")
    
    # 使用总样本的15%进行微调
    print(f"\n选择总样本的{config['fine_tune_percentage']*100:.1f}%进行OASIS样本微调...")
    total_size = len(oasis_brain_files)
    fine_tune_samples = max(1, int(total_size * config['fine_tune_percentage']))
    print(f"按15%计算的微调样本数: {fine_tune_samples}")
    
    # 使用分层随机采样选择固定比例的样本，保持原始类别比例
    if fine_tune_samples < total_size:
        # 计算每个类别的样本数
        from collections import Counter
        label_counts = Counter(oasis_labels)
        selected_indices = []
        
        # 对每个类别进行采样
        for label in set(oasis_labels):
            # 获取该类别的所有索引
            label_indices = np.where(np.array(oasis_labels) == label)[0]
            # 计算该类别应采样的比例
            label_ratio = label_counts[label] / total_size
            # 根据比例计算该类别应采样的数量，确保总和为fine_tune_samples
            samples_from_label = int(np.round(label_ratio * fine_tune_samples))
            # 确保不超过该类别的实际数量
            samples_from_label = min(samples_from_label, len(label_indices))
            # 随机选择样本
            if samples_from_label > 0:
                selected_indices.extend(random.sample(list(label_indices), samples_from_label))
        
        # 如果总样本数不足计算值，补充到计算值
        if len(selected_indices) < fine_tune_samples:
            remaining_indices = [i for i in range(total_size) if i not in selected_indices]
            additional_needed = fine_tune_samples - len(selected_indices)
            selected_indices.extend(random.sample(remaining_indices, min(additional_needed, len(remaining_indices))))
        
        train_val_indices = np.array(selected_indices)
    else:
        # 如果总样本数不足30，则使用所有样本
        train_val_indices = np.arange(total_size)
    
    fine_tune_size = len(train_val_indices)
    print(f"分层采样后实际使用样本数: {fine_tune_size}")
    
    # 对这15%的样本再次进行分层划分，保持类别比例
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.2,
        stratify=[oasis_labels[i] for i in train_val_indices],
        random_state=42
    )
    
    train_brain_files = [oasis_brain_files[i] for i in train_indices]
    train_hipp_files = [oasis_hipp_files[i] for i in train_indices]
    train_labels = [oasis_labels[i] for i in train_indices]
    
    val_brain_files = [oasis_brain_files[i] for i in val_indices]
    val_hipp_files = [oasis_hipp_files[i] for i in val_indices]
    val_labels = [oasis_labels[i] for i in val_indices]
    
    print(f"OASIS小比例数据划分完成:")
    print(f"  总样本数: {total_size}")
    print(f"  微调使用样本: {fine_tune_size} ({fine_tune_size/total_size*100:.1f}%)")
    print(f"  训练集大小: {len(train_brain_files)} 样本")
    print(f"  验证集大小: {len(val_brain_files)} 样本")
    
    # 分析类别分布
    def count_classes(indices, labels):
        selected_labels = [labels[i] for i in indices]
        class_counts = {0: selected_labels.count(0), 1: selected_labels.count(1), 2: selected_labels.count(2)}
        return class_counts
    
    original_classes = count_classes(np.arange(total_size), oasis_labels)
    train_classes = count_classes(train_indices, oasis_labels)
    val_classes = count_classes(val_indices, oasis_labels)
    
    print("\n类别分布分析:")
    print(f"  原始数据类别分布: {original_classes}")
    print(f"  训练集类别分布: {train_classes}")
    print(f"  验证集类别分布: {val_classes}")
    
    # 检查数据集大小
    if len(train_brain_files) < config['batch_size']:
        print(f"警告：训练集大小小于批次大小，调整批次大小为 {len(train_brain_files)}")
        config['batch_size'] = len(train_brain_files)
    
    if len(val_brain_files) < config['batch_size']:
        print(f"警告：验证集大小小于批次大小，调整批次大小为 {len(val_brain_files)}")
        config['batch_size'] = len(val_brain_files)
    
    # 创建数据集和数据加载器
    print("\n创建数据加载器...")
    train_dataset = EvalDataset(train_brain_files, train_hipp_files, train_labels)
    val_dataset = EvalDataset(val_brain_files, val_hipp_files, val_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows环境下设置为0
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 加载预训练模型
    print(f"\n加载预训练模型: {pretrained_model_path}")
    try:
        # 确保使用正确的模型类
        model = FullModel()
        
        # 加载预训练权重 - 处理分类头参数不匹配问题
        pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
        
        # 处理分类头参数不匹配问题
        # 由于我们在分类头中添加了Dropout层，需要单独处理分类头的参数
        model_dict = model.state_dict()
        
        # 过滤出不包含cls_head的参数
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('cls_head')}
        
        # 尝试加载除分类头外的所有参数
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        
        # 移动到设备
        model = model.to(config['device'])
        print("预训练模型加载成功！(分类头参数将重新初始化)")
        
        # 打印encoder层结构以验证第6~11层
        print("\n验证encoder层结构:")
        for i, layer in enumerate(model.encoder.encoder):
            print(f"Layer {i}: {layer.__class__.__name__}")
        
        # 重新初始化分类头 - 添加Dropout层
        class ClassificationHeadWithDropout(nn.Module):
            def __init__(self, input_dim=128, hidden_dim=512, num_classes=3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),  # 添加Dropout以减少过拟合
                    nn.Linear(hidden_dim, num_classes)
                )

            def forward(self, x):
                return self.net(x)
        
        # 使用配置中的feature_dim
        feature_dim = config['feature_dim'] if 'feature_dim' in config else 128
        # 创建新的分类头并确保它在与模型相同的设备上
        device = next(model.parameters()).device
        model.cls_head = ClassificationHeadWithDropout(input_dim=feature_dim, hidden_dim=512, num_classes=3).to(device)
        print(f"分类头参数已重新初始化为带Dropout的版本并移动到设备: {device}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 辅助函数：动态解冻层
    def set_requires_grad(model, layer_keywords, requires_grad=True):
        """按关键词控制指定层的 requires_grad"""
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in layer_keywords):
                param.requires_grad = requires_grad
    
    def freeze_bn_affine_only(model):
        """
        部分 BN 冻结策略：冻结 BatchNorm 层的 affine 参数（weight 和 bias），
        设置为评估模式不更新统计量，更符合"冻结encoder"的目标
        """
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.train()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def set_freeze_strategy(model):
        """设置统一的冻结策略：只解冻分类头，保持BatchNorm统计量更新
        
        参数:
            model: 模型实例
        """
        # 首先冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻分类头
        print("统一冻结策略：只解冻分类头")
        for param in model.cls_head.parameters():
            param.requires_grad = True
        
        # 保持所有BatchNorm层在训练模式并冻结affine参数
        freeze_bn_affine_only(model)
    
    print("\n参数冻结配置:")
    print("训练策略: 统一冻结策略")
    print("# 所有50轮：只解冻分类头")
    print("# 所有BatchNorm层保持在训练模式以更新统计量，但affine参数被冻结")
    print("分类头已添加Dropout(p=0.3)以减少过拟合")
    
    # 设置统一的冻结策略
    set_freeze_strategy(model)
    
    # 统计初始阶段的参数情况
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    print(f"\n参数统计 - 初始阶段:")
    print(f"  冻结参数: {frozen_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  总参数: {frozen_params + trainable_params:,}")
    print(f"  训练参数比例: {trainable_params/(frozen_params+trainable_params)*100:.2f}%")
    
    # 优化器 - 只优化分类头参数
    optimizer = torch.optim.Adam(
        model.cls_head.parameters(),
        lr=1e-4,
        weight_decay=config['weight_decay']
    )
    
    # 损失函数 - 添加标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # 早停机制 - 耐心值为10个epoch，监控验证损失防止过拟合
    early_stopping = EarlyStopping(
        patience=config['patience'],
        verbose=True,
        delta=0.0001
    )
    
    # 训练历史记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\n开始微调训练...")
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 统一的阶段描述
        phase_desc = "(只解冻分类头)"
        
        for (b, h), labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} {phase_desc}"):
            b, h, labels = map(lambda x: x.to(config['device']), [b, h, labels])
            
            # 前向传播
            optimizer.zero_grad()
            _, logits = model(b, h)
            loss = criterion(logits, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # 计算训练指标
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for (b, h), labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - 验证"):
                b, h, labels = map(lambda x: x.to(config['device']), [b, h, labels])
                
                # 前向传播
                _, logits = model(b, h)
                loss = criterion(logits, labels)
                
                # 统计
                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        # 计算验证指标
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # 打印epoch结果
        print(f"Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"  训练损失: {train_epoch_loss:.6f}, 训练准确率: {train_epoch_acc:.4f}")
        print(f"  验证损失: {val_epoch_loss:.6f}, 验证准确率: {val_epoch_acc:.4f}")
        
        # 早停检查
        early_stopping(val_epoch_loss, model, model_save_path)
        
        if early_stopping.early_stop:
            print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
            break
    
    # 加载最佳模型
    print(f"\n加载最佳微调模型")
    model.load_state_dict(torch.load(model_save_path, map_location=config['device']))
    
    # 在验证集上评估最佳模型
    print("\n在验证集上评估最佳模型...")
    model.eval()
    val_preds, val_true_labels, val_probs = [], [], []
    
    with torch.no_grad():
        for (b, h), labels in val_loader:
            b, h, labels = map(lambda x: x.to(config['device']), [b, h, labels])
            _, logits = model(b, h)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
            val_probs.append(probs.cpu().numpy())
    
    # 处理结果
    val_probs = np.concatenate(val_probs) if val_probs else np.array([])
    val_preds = np.array(val_preds)
    val_true_labels = np.array(val_true_labels)
    
    # 计算评估指标
    val_accuracy = accuracy_score(val_true_labels, val_preds)
    val_precision = precision_score(val_true_labels, val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_true_labels, val_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(val_true_labels, val_preds, average='weighted', zero_division=0)
    val_auc = compute_auc(val_true_labels, val_probs, "OASIS验证集")
    
    print(f"\n最佳模型验证结果:")
    print(f"  验证准确率: {val_accuracy:.4f}")
    print(f"  验证精确率: {val_precision:.4f}")
    print(f"  验证召回率: {val_recall:.4f}")
    print(f"  验证F1值: {val_f1:.4f}")
    print(f"  验证AUC: {val_auc:.4f}" if not np.isnan(val_auc) else "  验证AUC: NaN")
    
    # 保存微调结果
    with open(os.path.join(output_dir, "finetune_results.txt"), 'w') as f:
        f.write("OASIS微调结果\n")
        f.write(f"预训练模型: {pretrained_model_path}\n")
        f.write(f"\n训练配置:")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        
        f.write(f"\n训练过程:")
        for i in range(len(train_losses)):
            f.write(f"  Epoch {i+1}: 训练损失={train_losses[i]:.6f}, 训练准确率={train_accs[i]:.4f}, ")
            f.write(f"验证损失={val_losses[i]:.6f}, 验证准确率={val_accs[i]:.4f}\n")
        
        f.write(f"\n最佳模型验证结果:")
        f.write(f"  验证准确率: {val_accuracy:.4f}\n")
        f.write(f"  验证精确率: {val_precision:.4f}\n")
        f.write(f"  验证召回率: {val_recall:.4f}\n")
        f.write(f"  验证F1值: {val_f1:.4f}\n")
        f.write(f"  验证AUC: {val_auc:.4f}\n" if not np.isnan(val_auc) else "  验证AUC: NaN\n")
    
    print(f"\n微调完成！")
    print(f"最佳模型保存至: {model_save_path}")
    print(f"结果记录保存至: {os.path.join(output_dir, 'finetune_results.txt')}")
    

    
    return {
        'best_model_path': model_save_path,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'config': config
    }

if __name__ == "__main__":
    # 指定模型路径和数据路径
    MODEL_PATH = r"F:\ADNI\ClassificationAD\PROJECT\CONTRAST_LEARNING-master\runs\experiment_20251024_110747\fold_1\best_model.pth"
    
    # OASIS数据路径
    OASIS_BRAIN_ROOT = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\OAS\OAS_brain\OAS_brain_nyul"
    OASIS_HIPP_ROOT = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\OAS\OAS_hipp\Lside_resampled"
    
    # 结果保存路径
    RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    FINETUNE_DIR = os.path.join(RESULT_DIR, "finetuned2")
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FINETUNE_DIR, exist_ok=True)
    
    # 执行微调
    print("执行OASIS微调...")
    finetune_results = finetune_oasis_model(MODEL_PATH, OASIS_BRAIN_ROOT, OASIS_HIPP_ROOT, FINETUNE_DIR)
    
    # 检查微调是否成功
    if finetune_results is None:
        print("\n错误：OASIS微调出错，无法继续测试流程！")
    else:
        # 使用微调后的模型进行测试
        print("\n使用微调后的模型进行OASIS独立测试...")
        try:
            test_results = test_oasis_with_model(
                finetune_results['best_model_path'], 
                OASIS_BRAIN_ROOT, 
                OASIS_HIPP_ROOT, 
                RESULT_DIR
            )
            print("\nOASIS微调与测试流程完成！")
        except Exception as e:
            print(f"\n测试过程出错: {e}")
            print("微调已完成，但测试阶段失败")
    
    print("\n程序执行结束")