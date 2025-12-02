# import os
# from sklearn.model_selection import StratifiedKFold

# def get_fold_file_lists(brain_root, hipp_root):
#     """
#     读取脑数据文件列表和标签，根据脑数据命名推导海马文件名，
#     用 StratifiedKFold 做5折划分，返回每折的训练和验证路径及标签。

#     Args:
#         brain_root (str): 脑数据根目录（包含 AD, CN, MCI 子文件夹）
#         hipp_root (str): 海马数据根目录（对应脑数据结构）

#     Returns:
#         folds (list): 长度5，元素为dict，包含：
#             - train_brain_files, train_hipp_files, train_labels
#             - val_brain_files, val_hipp_files, val_labels
#     """
#     brain_files = []
#     labels = []

#     class_map = {"AD": 0, "CN": 1, "MCI": 2}



#     # 遍历脑数据文件，获取所有路径和标签
#     for cls in ["AD", "CN", "MCI"]:
#         cls_dir = os.path.join(brain_root, cls)
#         if not os.path.exists(cls_dir):
#             print(f"当前类别目录 {cls_dir} 不存在")
#             continue

#         # ✅ 这段必须在 for 循环里（不要顶格写）
#         for fname in os.listdir(cls_dir):
#             if not fname.endswith(".nii.gz"):
#                 continue
#             brain_files.append(os.path.join(cls_dir, fname))
#             labels.append(class_map[cls])

#                 # ✅ 检查标签加载是否正确
#     print("[DEBUG] 加载脑数据文件数:", len(brain_files))
#     print("[DEBUG] 标签分布:", Counter(labels))


#     # 推导对应海马文件名的函数
#     def get_hipp_file(brain_file):
#         base_name = os.path.basename(brain_file)
#         hipp_name = base_name.replace(".nii.gz", "_L_Hipp.nii.gz")
#         cls = brain_file.split(os.sep)[-2]
#         return os.path.join(hipp_root, cls, hipp_name)

#     # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)   # 暂时将5折改成2折

#     folds = []

#     for train_idx, val_idx in skf.split(brain_files, labels):
#         train_brain_files = [brain_files[i] for i in train_idx]
#         val_brain_files = [brain_files[i] for i in val_idx]

#         train_hipp_files = [get_hipp_file(f) for f in train_brain_files]
#         val_hipp_files = [get_hipp_file(f) for f in val_brain_files]

#         train_labels = [labels[i] for i in train_idx]
#         val_labels = [labels[i] for i in val_idx]

#         fold = {
#             "train_brain_files": train_brain_files,
#             "train_hipp_files": train_hipp_files,
#             "train_labels": train_labels,
#             "val_brain_files": val_brain_files,
#             "val_hipp_files": val_hipp_files,
#             "val_labels": val_labels
#         }
#         folds.append(fold)

#     return folds

import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# from config import brain_root
# from test_data_loading import hipp_root

def is_augmented(f):
    """检查文件是否为增强样本"""
    return os.path.basename(f).startswith(('blur_', 'flip_', 'spike_'))

def strip_prefix(fname):
    """移除文件名前缀"""
    for p in ['blur_', 'flip_', 'spike_']:
        if fname.startswith(p):
            return fname[len(p):]
    return fname

def get_original_base_name(file_path):
    """获取文件的原始基础名称（去除路径和增强前缀）"""
    fname = os.path.basename(file_path)
    return strip_prefix(fname)

def get_fold_file_lists(brain_root, hipp_root, n_splits=5, random_state=42, val_ratio=0.2):
    """
    按照5个步骤实现数据划分：
    1. 首先区分原图和增强图
    2. 将原图按照8-2分成训练和验证
    3. 给训练和验证的原图分别匹配上增强图
    4. 检查剩下的所有增强图数量
    5. 条件判断，如果剩下的增强图是来自训练集原图的增强，就放到训练集，否则放在验证集

    Args:
        brain_root (str): 脑数据根目录（包含 AD, CN, MCI 子文件夹）
        hipp_root (str): 海马数据根目录（对应脑数据结构）
        n_splits (int): 交叉验证折数
        random_state (int): 随机种子
        val_ratio (float): 验证集比例，默认0.2（8:2划分）

    Returns:
        folds (list): 长度为n_splits，元素为dict，包含：
            - train_brain_files, train_hipp_files, train_labels
            - val_brain_files, val_hipp_files, val_labels
    """
    all_brain_files = []
    all_hipp_files = []
    all_labels = []
    class_map = {"AD": 0, "CN": 1, "MCI": 2}

    # 遍历脑数据文件，获取所有路径和标签
    for cls in ["AD", "CN", "MCI"]:
        cls_dir = os.path.join(brain_root, cls)
        if not os.path.exists(cls_dir):
            # print(f"[WARNING] 当前类别目录 {cls_dir} 不存在")
            continue

        for fname in os.listdir(cls_dir):
            if not fname.endswith(('.nii.gz', '.nii')):
                continue
                
            brain_path = os.path.join(cls_dir, fname)
            # 推导对应海马文件名
            if fname.endswith('.nii.gz'):
                hipp_fname = fname.replace(".nii.gz", "_L_Hipp.nii.gz")
            else:
                hipp_fname = fname.replace("_brain_regist.nii", "_brain_regist_L_Hipp.nii")
            
            hipp_path = os.path.join(hipp_root, cls, hipp_fname)
            
            if os.path.exists(hipp_path):
                all_brain_files.append(brain_path)
                all_hipp_files.append(hipp_path)
                all_labels.append(class_map[cls])
            else:
                print(f"[警告] 找不到海马图像: {hipp_path}")

    # 检查标签加载是否正确
    # print("[DEBUG] 加载脑数据文件总数:", len(all_brain_files))
    # print("[DEBUG] 总标签分布:", Counter(all_labels))
    
    # 步骤1: 首先区分原图和增强图
    print("\n[步骤1] 区分原图和增强图...")
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
    
    print(f"[DEBUG] 原图数量: {len(orig_brain)}, 增强图数量: {len(aug_brain)}")
    print(f"[DEBUG] 原图标签分布: {Counter(orig_labels)}")
    print(f"[DEBUG] 增强图标签分布: {Counter(aug_labels)}")
    
    # 创建一个映射：原始基础名称 -> 所属集合（训练集或验证集）
    base_name_to_set = {}
    folds = []
    
    # 为每个折执行8:2划分
    for fold_idx in range(n_splits):
        print(f"\n[折 {fold_idx+1}/{n_splits}] 开始划分...")
        
        # 设置当前折的随机种子，确保不同折有不同的随机划分
        current_seed = random_state + fold_idx
        
        # 步骤2: 将原图按照8-2分成训练和验证
        # 参数验证和规范化val_ratio
        if not isinstance(val_ratio, (int, float)):
            raise TypeError("val_ratio must be int or float")
        
        # 规范化val_ratio参数为0-1范围内的浮点数
        if isinstance(val_ratio, int):
            # 如果是整数，转换为浮点数比例（假设是百分比）
            if val_ratio < 0:
                print("[警告] val_ratio为负数，使用默认值0.2")
                test_size = 0.2
            elif val_ratio > 100:
                print("[警告] val_ratio超过100，使用默认值0.2")
                test_size = 0.2
            elif val_ratio > 1:
                # 假设是百分比值
                test_size = val_ratio / 100.0
            else:
                test_size = float(val_ratio)
        else:
            # 如果是浮点数，确保在合理范围内
            if val_ratio <= 0 or val_ratio >= 1:
                print(f"[警告] val_ratio={val_ratio}超出(0,1)范围，使用默认值0.2")
                test_size = 0.2
            else:
                test_size = val_ratio
        
        print(f"[步骤2] 将原图按{int((1-test_size)*100)}:{int(test_size*100)}划分训练集和验证集...")
        # 使用分层抽样确保类别分布
        orig_train_b, orig_val_b, orig_train_h, orig_val_h, orig_train_y, orig_val_y = train_test_split(
            orig_brain, orig_hipp, orig_labels, test_size=test_size, 
            stratify=orig_labels, random_state=current_seed
        )
        
        # 更新基础名称映射
        for b_path in orig_train_b:
            base_name = get_original_base_name(b_path)
            base_name_to_set[base_name] = 'train'
        
        for b_path in orig_val_b:
            base_name = get_original_base_name(b_path)
            base_name_to_set[base_name] = 'val'
        
        print(f"[DEBUG] 训练集原图: {len(orig_train_b)}, 验证集原图: {len(orig_val_b)}")
        print(f"[DEBUG] 训练集原图标签分布: {Counter(orig_train_y)}")
        print(f"[DEBUG] 验证集原图标签分布: {Counter(orig_val_y)}")
        
        # 步骤3: 给训练和验证的原图分别匹配上增强图
        print("[步骤3] 为训练集和验证集原图匹配增强图...")
        train_brain_files = orig_train_b.copy()
        train_hipp_files = orig_train_h.copy()
        train_labels = orig_train_y.copy()
        
        val_brain_files = orig_val_b.copy()
        val_hipp_files = orig_val_h.copy()
        val_labels = orig_val_y.copy()
        
        # 步骤4和5: 检查所有增强图并分配到对应的集合
        print("[步骤4-5] 检查并分配所有增强图...")
        processed_aug_count = 0
        
        for b_path, h_path, label in zip(aug_brain, aug_hipp, aug_labels):
            # 获取增强图对应的原始基础名称
            base_name = get_original_base_name(b_path)
            
            # 条件判断：根据原图所属集合分配增强图
            if base_name in base_name_to_set:
                if base_name_to_set[base_name] == 'train':
                    train_brain_files.append(b_path)
                    train_hipp_files.append(h_path)
                    train_labels.append(label)
                # else:  # 'val'
                #     val_brain_files.append(b_path)
                #     val_hipp_files.append(h_path)
                #     val_labels.append(label)
                processed_aug_count += 1
        
        print(f"[DEBUG] 已处理增强图数量: {processed_aug_count}/{len(aug_brain)}")
        
        # 断言所有增强图都被正确处理
        assert processed_aug_count == len(aug_brain), \
            f"[错误] Fold {fold_idx+1} 存在未处理的增强图: {len(aug_brain) - processed_aug_count} 个"
        
        # 检查数量一致性
        assert len(train_brain_files) == len(train_hipp_files) == len(train_labels), \
            f"[错误] Fold {fold_idx+1} 训练集数量不一致"
        assert len(val_brain_files) == len(val_hipp_files) == len(val_labels), \
            f"[错误] Fold {fold_idx+1} 验证集数量不一致"
        
        # 计算训练集和验证集的划分比例
        total_samples = len(train_brain_files) + len(val_brain_files)
        train_ratio = len(train_brain_files) / total_samples * 100
        val_ratio = len(val_brain_files) / total_samples * 100
        
        print(f"[折 {fold_idx+1}] 最终划分结果:")
        print(f"[折 {fold_idx+1}] 训练集总数: {len(train_brain_files)}, 验证集总数: {len(val_brain_files)}")
        print(f"[折 {fold_idx+1}] 训练集比例: {train_ratio:.1f}%, 验证集比例: {val_ratio:.1f}%")
        print(f"[折 {fold_idx+1}] 训练集标签分布: {Counter(train_labels)}")
        print(f"[折 {fold_idx+1}] 验证集标签分布: {Counter(val_labels)}")
        
        fold = {
            "train_brain_files": train_brain_files,
            "train_hipp_files": train_hipp_files,
            "train_labels": train_labels,
            "val_brain_files": val_brain_files,
            "val_hipp_files": val_hipp_files,
            "val_labels": val_labels,
        }
        folds.append(fold)
        
        # 清空基础名称映射，为下一折做准备
        base_name_to_set.clear()

    return folds

if __name__ == "__main__":
    # 测试函数
    brain_root = r'F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\NIFTI5\all'
    hipp_root = r'F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\hippdata\all'
    folds = get_fold_file_lists(brain_root, hipp_root, n_splits=5, random_state=42, val_ratio=0.2)
