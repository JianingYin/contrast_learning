import os
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold


def is_augmented(file_path):
    """
    检查文件是否为增强样本
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        bool: 如果是增强样本返回True，否则返回False
    """
    return os.path.basename(file_path).startswith(('blur_', 'flip_', 'spike_'))


def strip_prefix(filename):
    """
    移除文件名前缀
    
    Args:
        filename (str): 文件名
    
    Returns:
        str: 移除前缀后的文件名
    """
    for prefix in ['blur_', 'flip_', 'spike_']:
        if filename.startswith(prefix):
            return filename[len(prefix):]
    return filename


def get_original_base_name(file_path):
    """
    获取文件的原始基础名称（去除路径和增强前缀）
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        str: 原始基础名称
    """
    filename = os.path.basename(file_path)
    return strip_prefix(filename)


def get_stratified_kfold_lists(brain_root, hipp_root, n_splits=5, random_state=42):
    """
    使用标准StratifiedKFold实现5折交叉验证数据划分，确保每个折中各类别比例一致
    
    Args:
        brain_root (str): 脑数据根目录（包含 AD, CN, MCI 子文件夹）
        hipp_root (str): 海马数据根目录（对应脑数据结构）
        n_splits (int): 交叉验证折数，默认为5
        random_state (int): 随机种子，确保结果可重现
    
    Returns:
        list: 长度为n_splits的列表，每个元素为字典，包含以下键：
            - train_brain_files: 训练集脑数据文件路径列表
            - train_hipp_files: 训练集海马数据文件路径列表
            - train_labels: 训练集标签列表
            - val_brain_files: 验证集脑数据文件路径列表
            - val_hipp_files: 验证集海马数据文件路径列表
            - val_labels: 验证集标签列表
    """
    # 初始化空列表存储所有文件路径和标签
    all_brain_files = []
    all_hipp_files = []
    all_labels = []
    
    # 定义类别映射
    class_map = {"AD": 0, "CN": 1, "MCI": 2}
    
    # 遍历脑数据文件，获取所有路径和标签
    for cls in ["AD", "CN", "MCI"]:
        cls_dir = os.path.join(brain_root, cls)
        if not os.path.exists(cls_dir):
            print(f"[WARNING] 当前类别目录 {cls_dir} 不存在，跳过该类别")
            continue
        
        for filename in os.listdir(cls_dir):
            if not filename.endswith(('.nii.gz', '.nii')):
                continue
                
            brain_path = os.path.join(cls_dir, filename)
            # 推导对应海马文件名
            if filename.endswith('.nii.gz'):
                hipp_filename = filename.replace(".nii.gz", "_L_Hipp.nii.gz")
            else:
                hipp_filename = filename.replace("_brain_regist.nii", "_brain_regist_L_Hipp.nii")
            
            hipp_path = os.path.join(hipp_root, cls, hipp_filename)
            
            if os.path.exists(hipp_path):
                all_brain_files.append(brain_path)
                all_hipp_files.append(hipp_path)
                all_labels.append(class_map[cls])
            else:
                print(f"[WARNING] 找不到海马图像: {hipp_path}")
    
    # 检查标签加载是否正确
    print(f"[INFO] 加载脑数据文件总数: {len(all_brain_files)}")
    print(f"[INFO] 总标签分布: {Counter(all_labels)}")
    
    # 步骤1: 区分原图和增强图
    print(f"\n[步骤1] 区分原图和增强图...")
    # 找出所有非增强图（原图）
    non_aug_indices = [i for i, f in enumerate(all_brain_files) if not is_augmented(f)]
    orig_brain = [all_brain_files[i] for i in non_aug_indices]
    orig_hipp = [all_hipp_files[i] for i in non_aug_indices]
    orig_labels = [all_labels[i] for i in non_aug_indices]
    
    # 找出所有增强图
    aug_indices = [i for i, f in enumerate(all_brain_files) if is_augmented(f)]
    aug_brain = [all_brain_files[i] for i in aug_indices]
    aug_hipp = [all_hipp_files[i] for i in aug_indices]
    aug_labels = [all_labels[i] for i in aug_indices]
    
    print(f"[INFO] 原图数量: {len(orig_brain)}, 增强图数量: {len(aug_brain)}")
    print(f"[INFO] 原图标签分布: {Counter(orig_labels)}")
    print(f"[INFO] 增强图标签分布: {Counter(aug_labels)}")
    
    # 初始化折列表
    folds = []
    
    # 使用StratifiedKFold进行标准的5折交叉验证划分
    print(f"\n[INFO] 使用StratifiedKFold进行{n_splits}折交叉验证划分...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 迭代每个折
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(orig_brain, orig_labels)):
        print(f"\n[折 {fold_idx+1}/{n_splits}] 开始划分...")
        
        # 使用StratifiedKFold提供的索引直接划分数据
        orig_train_b = [orig_brain[i] for i in train_idx]
        orig_train_h = [orig_hipp[i] for i in train_idx]
        orig_train_y = [orig_labels[i] for i in train_idx]
        
        orig_val_b = [orig_brain[i] for i in val_idx]
        orig_val_h = [orig_hipp[i] for i in val_idx]
        orig_val_y = [orig_labels[i] for i in val_idx]
        
        print(f"[步骤2] 已完成原图的{len(orig_train_b)}:{len(orig_val_b)}训练集和验证集划分")
        print(f"[INFO] 训练集原图标签分布: {Counter(orig_train_y)}")
        print(f"[INFO] 验证集原图标签分布: {Counter(orig_val_y)}")
        
        # 创建一个映射：原始基础名称 -> 所属集合（训练集或验证集）
        base_name_to_set = {}
        
        # 更新基础名称映射
        for b_path in orig_train_b:
            base_name = get_original_base_name(b_path)
            base_name_to_set[base_name] = 'train'
        
        for b_path in orig_val_b:
            base_name = get_original_base_name(b_path)
            base_name_to_set[base_name] = 'val'
        
        # 步骤3: 为训练集原图匹配增强图，验证集只保留原图
        print("[步骤3] 为训练集原图匹配增强图，验证集只保留原图...")
        train_brain_files = orig_train_b.copy()
        train_hipp_files = orig_train_h.copy()
        train_labels = orig_train_y.copy()
        
        # 验证集只保留原图
        val_brain_files = orig_val_b.copy()
        val_hipp_files = orig_val_h.copy()
        val_labels = orig_val_y.copy()
        
        # 步骤4: 检查并分配增强图到训练集
        print("[步骤4] 检查并分配增强图到训练集...")
        processed_aug_count = 0
        
        for b_path, h_path, label in zip(aug_brain, aug_hipp, aug_labels):
            # 获取增强图对应的原始基础名称
            base_name = get_original_base_name(b_path)
            
            # 只将增强图分配到训练集
            if base_name in base_name_to_set:
                if base_name_to_set[base_name] == 'train':
                    train_brain_files.append(b_path)
                    train_hipp_files.append(h_path)
                    train_labels.append(label)
                    processed_aug_count += 1
                # 对于验证集，不添加增强图
        
        # 计算未处理的增强图数量
        unprocessed_count = len(aug_brain) - processed_aug_count
        print(f"[INFO] 已处理增强图数量: {processed_aug_count}/{len(aug_brain)}")
        print(f"[INFO] 未处理的验证集增强图数量: {unprocessed_count}")
        
        # 验证训练集只包含训练集原图的增强图，不包含验证集原图的增强图
        print("[步骤5] 验证增强图分配正确性...")
        
        # 检查1: 确保所有训练集原图的增强图都被添加到训练集中
        missing_train_aug_count = 0
        for orig_path in orig_train_b:
            orig_base = get_original_base_name(orig_path)
            # 查找该原图对应的所有增强图是否都在训练集中
            for aug_path, aug_h_path, label in zip(aug_brain, aug_hipp, aug_labels):
                aug_base = get_original_base_name(aug_path)
                if aug_base == orig_base and aug_path not in train_brain_files:
                    missing_train_aug_count += 1
                    print(f"[WARNING] 训练集原图 {orig_base} 的增强图 {os.path.basename(aug_path)} 未被添加到训练集")
        
        # 检查2: 确保没有验证集原图的增强图被添加到训练集
        val_aug_in_train_count = 0
        for orig_path in orig_val_b:
            orig_base = get_original_base_name(orig_path)
            for aug_path in train_brain_files:
                # 只检查训练集中的增强图
                if is_augmented(aug_path):
                    aug_base = get_original_base_name(aug_path)
                    if aug_base == orig_base:
                        val_aug_in_train_count += 1
                        print(f"[ERROR] 验证集原图 {orig_base} 的增强图 {os.path.basename(aug_path)} 被错误添加到训练集")
        
        # 断言检查
        assert missing_train_aug_count == 0, f"[ERROR] 折 {fold_idx+1} 有 {missing_train_aug_count} 个训练集原图的增强图未被添加到训练集"
        assert val_aug_in_train_count == 0, f"[ERROR] 折 {fold_idx+1} 有 {val_aug_in_train_count} 个验证集原图的增强图被错误添加到训练集"
        
        print(f"[INFO] 增强图分配检查通过，所有训练集原图的增强图都已添加到训练集，且没有验证集原图的增强图被添加")
        
        # 检查数量一致性
        assert len(train_brain_files) == len(train_hipp_files) == len(train_labels), \
            f"[ERROR] 折 {fold_idx+1} 训练集数量不一致"
        assert len(val_brain_files) == len(val_hipp_files) == len(val_labels), \
            f"[ERROR] 折 {fold_idx+1} 验证集数量不一致"
        
        # 计算训练集和验证集的划分比例
        total_samples = len(train_brain_files) + len(val_brain_files)
        train_ratio = len(train_brain_files) / total_samples * 100
        val_ratio = len(val_brain_files) / total_samples * 100
        
        print(f"[折 {fold_idx+1}] 最终划分结果:")
        print(f"[折 {fold_idx+1}] 训练集总数: {len(train_brain_files)}, 验证集总数: {len(val_brain_files)}")
        print(f"[折 {fold_idx+1}] 训练集比例: {train_ratio:.1f}%, 验证集比例: {val_ratio:.1f}%")
        print(f"[折 {fold_idx+1}] 训练集标签分布: {Counter(train_labels)}")
        print(f"[折 {fold_idx+1}] 验证集标签分布: {Counter(val_labels)}")
        
        # 构建当前折的结果字典
        fold = {
            "train_brain_files": train_brain_files,
            "train_hipp_files": train_hipp_files,
            "train_labels": train_labels,
            "val_brain_files": val_brain_files,
            "val_hipp_files": val_hipp_files,
            "val_labels": val_labels,
        }
        folds.append(fold)
    
    return folds


if __name__ == "__main__":
    # 测试函数
    brain_root = r'F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\NIFTI5\all'
    hipp_root = r'F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\hippdata\all'
    folds = get_stratified_kfold_lists(brain_root, hipp_root, n_splits=5, random_state=42)
    print(f"\n[INFO] 成功生成 {len(folds)} 折交叉验证数据")