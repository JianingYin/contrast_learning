#  交叉验证对应的数据划分方法
#================================================================================================================================================================================#

# import os
# import nibabel as nib
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class PairedNiftiDataset(Dataset):
#     def __init__(self, brain_dir=None, hippocampus_dir=None,
#                  brain_files=None, hipp_files=None, labels=None,
#                  transform=None):
#         """
#         支持两种初始化方式：
#         1) 传入 brain_dir 和 hippocampus_dir，自动扫描目录构造样本
#         2) 传入 brain_files, hipp_files, labels，直接用列表构造样本（5折划分时用）

#         Args:
#             brain_dir (str): 脑数据根目录（按类别子文件夹组织）
#             hippocampus_dir (str): 海马数据根目录（按类别子文件夹组织）
#             brain_files (list[str]): 脑数据文件完整路径列表
#             hipp_files (list[str]): 海马数据文件完整路径列表
#             labels (list[int]): 标签列表（数值形式）
#             transform (callable): 可选的图像预处理函数
#         """
#         self.transform = transform
#         self.samples = []
#         self.class_to_idx = {}

#         if brain_files is not None and hipp_files is not None and labels is not None:
#             # 直接用传入的列表构造样本
#             assert len(brain_files) == len(hipp_files) == len(labels), "文件和标签数量不匹配"
#             self.samples = list(zip(brain_files, hipp_files, labels))
#         elif brain_dir is not None and hippocampus_dir is not None:
#             # 通过扫描文件夹构造样本（老逻辑）
#             self.data_B_dir = brain_dir
#             self.data_h_dir = hippocampus_dir
#             self._build_dataset()
#         elif brain_files is not None and hipp_files is not None and labels is not None:
#             # 新逻辑：直接使用文件列表
#             self.brain_files = brain_files
#             self.hipp_files = hipp_files
#             self.labels = labels
#         else:
#             raise ValueError("必须传入 brain_dir 和 hippocampus_dir，或者 brain_files, hipp_files, labels")

#     def _build_dataset(self):
#         if hasattr(self, 'brain_files'):
#             # 处理文件列表
#             unique_classes = set([os.path.basename(os.path.dirname(f)) for f in self.brain_files])
#             self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
#         else:
#             # 老逻辑
#             classes = sorted(os.listdir(self.data_B_dir))
#             self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

#         for cls in classes:
#             b_cls_path = os.path.join(self.data_B_dir, cls)
#             h_cls_path = os.path.join(self.data_h_dir, cls)
#             if not os.path.isdir(b_cls_path) or not os.path.isdir(h_cls_path):
#                 continue

#             b_files = [f for f in os.listdir(b_cls_path) if f.endswith('.nii') or f.endswith('.nii.gz')]

#             for b_file in b_files:
#                 b_path = os.path.join(b_cls_path, b_file)
#                 h_file = self._to_hipp_name(b_file)
#                 h_path = os.path.join(h_cls_path, h_file)

#                 if os.path.exists(h_path):
#                     label_idx = self.class_to_idx[cls]
#                     self.samples.append((b_path, h_path, label_idx))
#                 else:
#                     print(f"[警告] 缺失对应的 hippocampus 文件: {h_path}")

#     def _to_hipp_name(self, b_file):
#         if b_file.endswith('.nii.gz'):
#             base = b_file[:-7]
#             return base + '_L_Hipp.nii.gz'
#         elif b_file.endswith('.nii'):
#             base = b_file[:-4]
#             return base + '_L_Hipp.nii'
#         else:
#             raise ValueError(f"不支持的文件扩展名: {b_file}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         b_path, h_path, label = self.samples[idx]

#         b_img = nib.load(b_path).get_fdata()
#         h_img = nib.load(h_path).get_fdata()

#         if b_img.shape != h_img.shape:
#             raise ValueError(f"[尺寸错误] {b_path} 与 {h_path} 尺寸不一致")

#         b_img = self._normalize(b_img).astype(np.float32)
#         h_img = self._normalize(h_img).astype(np.float32)

#         b_tensor = torch.from_numpy(b_img).unsqueeze(0)
#         h_tensor = torch.from_numpy(h_img).unsqueeze(0)

#         if self.transform:
#             b_tensor = self.transform(b_tensor)
#             h_tensor = self.transform(h_tensor)

#         return b_tensor, h_tensor, label

#     def _normalize(self, arr):
#         arr = np.nan_to_num(arr)
#         if np.std(arr) == 0:
#             return arr
#         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
#  交叉验证对应的数据划分方法
#================================================================================================================================================================================#



import os
import random
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# 似乎没有完全实现对比学习正样本的组建。前36个实验都用的是这个
# class PairedNiftiDataset(Dataset):
#     def __init__(self, brain_files, hipp_files, labels, transform=None):
#         self.brain_files = brain_files
#         self.hipp_files = hipp_files
#         self.labels = labels
#         self.transform = transform

#         assert len(self.brain_files) == len(self.hipp_files) == len(self.labels), "数据长度不一致"

#         # ✅ 构建样本元组列表
#         self.samples = list(zip(self.brain_files, self.hipp_files, self.labels))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         b_path, h_path, label = self.samples[idx]

#         b_img = nib.load(b_path).get_fdata()
#         h_img = nib.load(h_path).get_fdata()

#         if b_img.shape != h_img.shape:
#             raise ValueError(f"[尺寸错误] {b_path} 与 {h_path} 尺寸不一致")

#         # 归一化
#         b_img = self._normalize(b_img).astype(np.float32)
#         h_img = self._normalize(h_img).astype(np.float32)

#         # 加入通道维度：[1, D, H, W]
#         b_tensor = torch.from_numpy(b_img).unsqueeze(0)
#         h_tensor = torch.from_numpy(h_img).unsqueeze(0)

#         if self.transform:
#             b_tensor = self.transform(b_tensor)
#             h_tensor = self.transform(h_tensor)

#         return b_tensor, h_tensor, label

#     def _normalize(self, arr):
#         arr = np.nan_to_num(arr)
#         if np.std(arr) == 0:
#             return arr
#         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


# class PairedContrastiveDataset(Dataset):
#     def __init__(self, brain_files, hipp_files, labels, transform=None):
#         """
#         brain_files: List of all brain file paths (包含原始和增强样本)
#         hipp_files:  同上
#         labels:      与每对 brain/hipp 文件对应的标签
#         """
#         assert len(brain_files) == len(hipp_files) == len(labels)
#         self.samples = list(zip(brain_files, hipp_files, labels))
#         self.transform = transform

#         # 预处理：建立增强映射（根据你的命名规则）
#         self.sample_dict = {}  # 原始样本名 → 所有对应样本（含原始+增强）
#         for b_path, h_path, label in self.samples:
#             base_name = self._strip_prefix(os.path.basename(b_path))
#             self.sample_dict.setdefault(base_name, []).append((b_path, h_path, label))

#         # 只保留原始样本名作为 anchor
#         self.anchor_keys = [k for k in self.sample_dict if not k.startswith(('blur_', 'flip_', 'spike_'))]

#     def __len__(self):
#         return len(self.anchor_keys)

#     def __getitem__(self, idx):
#         anchor_key = self.anchor_keys[idx]
#         candidates = self.sample_dict[anchor_key]

#         # 如果只有1个视图，就复制自己
#         if len(candidates) == 1:
#             sample1 = sample2 = candidates[0]
#         else:
#             sample1, sample2 = random.sample(candidates, 2)

#         b1, h1, label1 = self._load_sample(sample1)
#         b2, h2, label2 = self._load_sample(sample2)

#         assert label1 == label2, f"增强样本标签不一致：{label1} vs {label2}"

#         return (b1, h1), (b2, h2), label1

#     def _load_sample(self, sample):
#         b_path, h_path, label = sample
#         b_img = self._normalize(nib.load(b_path).get_fdata()).astype(np.float32)
#         h_img = self._normalize(nib.load(h_path).get_fdata()).astype(np.float32)
#         b_tensor = torch.from_numpy(b_img).unsqueeze(0)
#         h_tensor = torch.from_numpy(h_img).unsqueeze(0)
#         if self.transform:
#             b_tensor = self.transform(b_tensor)
#             h_tensor = self.transform(h_tensor)
#         return b_tensor, h_tensor, label

#     def _normalize(self, arr):
#         arr = np.nan_to_num(arr)
#         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8) if np.std(arr) != 0 else arr

#     def _strip_prefix(self, fname):
#         # 去除 blur_ / flip_ / spike_
#         for p in ['blur_', 'flip_', 'spike_']:
#             if fname.startswith(p):
#                 return fname[len(p):]
#         return fname

# 训练集：包含原图和增强图，支持 SupConLoss 正样本对构建。

# 验证集：仅保留原图，使用 CrossEntropyLoss 进行评估。

# 内置了划分函数 split_train_val()，默认按原图 8:2 stratify 划分。

# 提供 PairedContrastiveDataset（用于训练）和 EvalDataset（用于验证）两种 Dataset 类。
import os
import glob
import random
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict


class PairedContrastiveDataset(Dataset):
    def __init__(self, brain_files, hipp_files, labels, transform=None):
        assert len(brain_files) == len(hipp_files) == len(labels)
        self.samples = list(zip(brain_files, hipp_files, labels))
        self.transform = transform

        # 创建 base_name 映射：去掉前缀后的原图名 → 所有视图样本
        self.sample_dict = defaultdict(list)
        for b_path, h_path, label in self.samples:
            base_name = self._strip_prefix(os.path.basename(b_path))
            self.sample_dict[base_name].append((b_path, h_path, label))

        # 只保留原图作为 anchor
        self.anchor_keys = [k for k in self.sample_dict if not k.startswith(('blur_', 'flip_', 'spike_'))]

    def __len__(self):
        return len(self.anchor_keys)

    def __getitem__(self, idx):
        anchor_key = self.anchor_keys[idx]
        candidates = self.sample_dict[anchor_key]

        if len(candidates) == 1:
            sample1 = sample2 = candidates[0]
        else:
            sample1, sample2 = random.sample(candidates, 2)

        b1, h1, label1 = self._load_sample(sample1)
        b2, h2, label2 = self._load_sample(sample2)

        assert label1 == label2, f"标签不一致：{label1} vs {label2}"
        return (b1, h1), (b2, h2), label1

    def _load_sample(self, sample):
        b_path, h_path, label = sample
        b_img = self._normalize(nib.load(b_path).get_fdata()).astype(np.float32)
        h_img = self._normalize(nib.load(h_path).get_fdata()).astype(np.float32)
        b_tensor = torch.from_numpy(b_img).unsqueeze(0)
        h_tensor = torch.from_numpy(h_img).unsqueeze(0)
        if self.transform:
            b_tensor = self.transform(b_tensor)
            h_tensor = self.transform(h_tensor)
        return b_tensor, h_tensor, label

    def _normalize(self, arr):
        arr = np.nan_to_num(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8) if np.std(arr) != 0 else arr

    def _strip_prefix(self, fname):
        for p in ['blur_', 'flip_', 'spike_']:
            if fname.startswith(p):
                return fname[len(p):]
        return fname


class EvalDataset(Dataset):
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


def collect_file_label_pairs(brain_root, hipp_root):
    brain_paths, hipp_paths, labels = [], [], []
    label_map = {'AD': 0, 'CN': 1, 'MCI': 2}

    for label_name, label_value in label_map.items():
        brain_dir = os.path.join(brain_root, label_name)
        hipp_dir = os.path.join(hipp_root, label_name)

        brain_files = sorted(glob.glob(os.path.join(brain_dir, '*.nii*')))

        for b_path in brain_files:
            fname = os.path.basename(b_path)
            h_path = os.path.join(hipp_dir, fname.replace('_brain_regist.nii', '_brain_regist_L_Hipp.nii'))
            if os.path.exists(h_path):
                brain_paths.append(b_path)
                hipp_paths.append(h_path)
                labels.append(label_value)
            else:
                print(f"[警告] 找不到海马图像: {h_path}")

    print("数据总计:")
    print("  brain:", len(brain_paths))
    print("  hipp:", len(hipp_paths))
    print("  label:", len(labels))
    return brain_paths, hipp_paths, labels


def strip_prefix(fname):
    for p in ['blur_', 'flip_', 'spike_']:
        if fname.startswith(p):
            return fname[len(p):]
    return fname


def is_augmented(f):
    return os.path.basename(f).startswith(('blur_', 'flip_', 'spike_'))


def collect_augmented_pairs(base_brain_list, all_brain_list, all_hipp_list, all_labels):
    """
    根据 base_brain_list 的 basename，查找所有与之匹配的增强样本对（brain + hipp），返回配对好的文件列表与标签。
    """
    base_names = set([os.path.basename(f) for f in base_brain_list])

    paired_brain = []
    paired_hipp = []
    paired_labels = []

    for b_path, h_path, label in zip(all_brain_list, all_hipp_list, all_labels):
        b_name = os.path.basename(b_path)
        if strip_prefix(b_name) in base_names:
            paired_brain.append(b_path)
            paired_hipp.append(h_path)
            paired_labels.append(label)

    return paired_brain, paired_hipp, paired_labels


def split_train_val(brain_paths, hipp_paths, labels, test_ratio=0.2, random_state=42):
    """
    划分训练集和验证集，并配对增强图像（仅训练集）
    """
    # Step 1: 找出所有非增强图（原图）
    non_aug_indices = [i for i, f in enumerate(brain_paths) if not is_augmented(f)]
    brain_orig = [brain_paths[i] for i in non_aug_indices]
    hipp_orig = [hipp_paths[i] for i in non_aug_indices]
    label_orig = [labels[i] for i in non_aug_indices]

    # Step 2: 用原图划分训练/验证
    train_b, val_b, train_h, val_h, train_y, val_y = train_test_split(
        brain_orig, hipp_orig, label_orig,
        test_size=test_ratio, stratify=label_orig, random_state=random_state
    )

    # Step 3: 为训练集找出所有匹配的增强图 + 原图（完整训练集）
    train_b_aug, train_h_aug, train_y_aug = collect_augmented_pairs(train_b, brain_paths, hipp_paths, labels)

    # 验证集只保留原图
    val_b_aug = val_b
    val_h_aug = val_h
    val_y_aug = val_y

    # Step 4: 最后长度一致性检查
    assert len(train_b_aug) == len(train_h_aug) == len(train_y_aug), "[增强配对错误] train集增强图数量不一致"
    assert len(val_b_aug) == len(val_h_aug) == len(val_y_aug), "[增强配对错误] val集图数量不一致"

    return train_b_aug, train_h_aug, train_y_aug, val_b_aug, val_h_aug, val_y_aug




