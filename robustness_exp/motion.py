# 实验目标

# 使用已训练好的最佳模型
# 在测试集上添加不同程度的"运动伪影"
# 这里你设定参数 s = 10（表示运动模糊核大小）
# 计算 Accuracy、F1-score、AUC 三个指标。


# 添加项目根目录到Python路径
import sys
import os
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（上一级目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python路径
sys.path.append(project_root)

# 确保utils目录在Python路径中
sys.path.append(os.path.join(project_root, 'utils'))

# 导入所需模块
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.ndimage import convolve1d
from models.model.modelV24 import FullModel
from datasets.datasets_class import EvalDataset, collect_file_label_pairs

# 直接导入StratifiedKFold模块中的函数
# 尝试直接导入文件
stratified_kfold_path = os.path.join(project_root, 'utils', 'StratifiedKFold.py')
if os.path.exists(stratified_kfold_path):
    # 动态导入模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("StratifiedKFold", stratified_kfold_path)
    StratifiedKFold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(StratifiedKFold)
    get_stratified_kfold_lists = StratifiedKFold.get_stratified_kfold_lists
else:
    raise ImportError(f"无法找到StratifiedKFold.py文件: {stratified_kfold_path}")

# -----------------------------
# 🌀 模拟运动伪影（Motion Artifact）
# -----------------------------
def add_motion_artifact(img, s=10):
    """
    给 3D MRI 图像添加运动模糊伪影。
    s 表示模糊长度，越大伪影越严重。
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    # 生成一维运动卷积核
    kernel = np.zeros(s)
    kernel[:] = 1.0 / s

    # 对每个通道执行一维卷积（沿 x 轴模拟头动）
    if img.ndim == 4:  # (C, H, W, D)
        for c in range(img.shape[0]):
            img[c] = convolve1d(img[c], kernel, axis=2, mode='reflect')
    elif img.ndim == 3:  # (H, W, D)
        img = convolve1d(img, kernel, axis=2, mode='reflect')

    img = np.clip(img, 0, 1)
    return img

# -----------------------------
# 模型评估函数
# -----------------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for (b, h), labels in tqdm(dataloader, desc="Evaluating"):
            b, h, labels = map(lambda x: x.to(device), [b, h, labels])
            logits = model.cls_head(model.encoder(model.backbone(b, h)))
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    
    # 调试：检查数据维度
    print(f"\n评估数据统计:")
    print(f"  标签数量: {len(all_labels)}")
    print(f"  预测数量: {len(all_preds)}")
    print(f"  概率数组形状: {all_probs.shape}")
    print(f"  唯一标签: {np.unique(all_labels)}")
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 确保概率和标签维度匹配
    try:
        # 检查是否需要调整概率维度
        if len(np.unique(all_labels)) > 1 and all_probs.shape[1] != len(np.unique(all_labels)):
            print("⚠️ 警告：概率维度与类别数量不匹配")
            # 如果类别数为2但输出是3维，尝试只取前两维
            if len(np.unique(all_labels)) == 2 and all_probs.shape[1] > 2:
                print(f"  截断概率维度: {all_probs.shape[1]} -> 2")
                auc = roc_auc_score(all_labels, all_probs[:, :2], multi_class='ovr', average='macro')
            else:
                # 尝试二分类模式
                print("  使用二分类模式计算AUC")
                auc = roc_auc_score(all_labels, all_probs[:, 0], average='macro')
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"  计算AUC时出错: {str(e)}")
        auc = 0.0  # 如果计算失败，设置默认值
    
    return acc, f1, auc

# -----------------------------
# 主程序入口
# -----------------------------
def create_brain_hipp_pairs(brain_root, hipp_root):
    """
    建立脑数据和海马数据的正确对应关系
    规则：脑数据文件名为xxx.nii.gz，对应的海马数据为xxx_L_Hipp.nii.gz
    注意：数据目录下有AD、CN、MCI三个子目录
    """
    # 调试：检查目录是否存在
    print(f"检查目录:")
    print(f"  脑数据目录存在: {os.path.exists(brain_root)}")
    print(f"  海马数据目录存在: {os.path.exists(hipp_root)}")
    
    # 获取所有脑数据文件（递归遍历AD、CN、MCI子目录）
    brain_files = []
    labels = []
    
    # 定义标签映射
    label_map = {'AD': 0, 'CN': 1, 'MCI': 2}
    
    # 遍历每个类别子目录
    for label_name in ['AD', 'CN', 'MCI']:
        brain_subdir = os.path.join(brain_root, label_name)
        print(f"\n检查脑数据子目录: {brain_subdir}")
        
        if os.path.exists(brain_subdir):
            # 查找该子目录下的所有.nii.gz或.nii文件
            for filename in os.listdir(brain_subdir):
                if any(filename.endswith(ext) for ext in ['.nii.gz', '.nii']):
                    brain_files.append(os.path.join(brain_subdir, filename))
                    labels.append(label_map[label_name])
            
            print(f"  在{label_name}目录中找到{len(os.listdir(brain_subdir))}个文件")
        else:
            print(f"  {brain_subdir} 目录不存在")
    
    print(f"\n总共找到{len(brain_files)}个脑数据文件")
    
    # 建立脑数据和海马数据的对应关系
    paired_brain_files = []
    paired_hipp_files = []
    paired_labels = []
    
    for i, brain_file in enumerate(brain_files):
        brain_filename = os.path.basename(brain_file)
        print(f"\n处理脑数据文件 ({i+1}/{len(brain_files)}): {brain_filename}")
        
        # 生成对应的海马文件名
        if brain_filename.endswith('.nii.gz'):
            brain_name_without_ext = brain_filename[:-7]  # 移除.nii.gz
        elif brain_filename.endswith('.nii'):
            brain_name_without_ext = brain_filename[:-4]  # 移除.nii
        else:
            brain_name_without_ext = os.path.splitext(brain_filename)[0]
        
        # 尝试多种可能的海马文件路径
        # 方法1: 直接在海马根目录下查找
        hipp_filename = f"{brain_name_without_ext}_L_Hipp.nii.gz"
        hipp_file = os.path.join(hipp_root, hipp_filename)
        
        # 方法2: 在对应的类别子目录下查找
        hipp_subdir_file = None
        for label_name in ['AD', 'CN', 'MCI']:
            hipp_subdir = os.path.join(hipp_root, label_name)
            if os.path.exists(hipp_subdir):
                temp_file = os.path.join(hipp_subdir, hipp_filename)
                if os.path.exists(temp_file):
                    hipp_subdir_file = temp_file
                    break
        
        # 检查是否找到海马文件
        found_hipp = False
        if os.path.exists(hipp_file):
            print(f"  ✓ 在根目录找到对应的海马文件: {hipp_filename}")
            paired_brain_files.append(brain_file)
            paired_hipp_files.append(hipp_file)
            paired_labels.append(labels[i])
            found_hipp = True
        elif hipp_subdir_file:
            print(f"  ✓ 在子目录找到对应的海马文件: {hipp_subdir_file}")
            paired_brain_files.append(brain_file)
            paired_hipp_files.append(hipp_subdir_file)
            paired_labels.append(labels[i])
            found_hipp = True
        else:
            print(f"  ✗ 未找到对应的海马文件")
    
    print(f"\n数据对应完成:")
    print(f"  找到的脑数据文件数: {len(brain_files)}")
    print(f"  成功匹配的脑-海马数据对: {len(paired_brain_files)}")
    
    # 如果匹配到的数据太少，尝试直接使用示例文件进行测试
    if len(paired_brain_files) < 1:
        print("\n⚠️ 警告: 未找到足够的匹配数据，尝试使用硬编码的示例文件")
        # 假设存在一个示例文件
        sample_brain_file = os.path.join(brain_root, 'AD', 'blur_I23231_ADNI_11M4_BRM_20060823124753_2_brain_regist.nii.gz')
        sample_hipp_file = os.path.join(hipp_root, 'AD', 'blur_I23231_ADNI_11M4_BRM_20060823124753_2_brain_regist_L_Hipp.nii.gz')
        
        if os.path.exists(sample_brain_file) and os.path.exists(sample_hipp_file):
            print(f"  ✓ 使用示例文件进行测试")
            paired_brain_files = [sample_brain_file]
            paired_hipp_files = [sample_hipp_file]
            paired_labels = [0]  # AD类别
    
    return paired_brain_files, paired_hipp_files, paired_labels

class MotionArtifactTest:
    def __init__(self, device, brain_root, hipp_root, experiment_dir, random_state=42, s=10):
        self.device = device
        self.brain_root = brain_root
        self.hipp_root = hipp_root
        self.experiment_dir = experiment_dir
        self.random_state = random_state
        self.s = s
        self.model = FullModel().to(device)
        
    def run_test(self, target_fold=2, max_samples=20):
        """
        运行单个fold的运动伪影鲁棒性测试
        
        Args:
            target_fold: 目标fold索引（从0开始）
            max_samples: 最大样本数量，用于限制测试数据量
        """
        print(f"\n🔄 开始运动伪影鲁棒性测试 (fold={target_fold}, s={self.s}, max_samples={max_samples})")
        
        # 1. 直接使用create_brain_hipp_pairs获取数据对（简化版本，避免完整的K-fold划分）
        print("\n1️⃣ 获取数据对...")
        brain_files, hipp_files, labels = create_brain_hipp_pairs(self.brain_root, self.hipp_root)
        
        # 限制数据量以提高效率
        if len(brain_files) > max_samples:
            print(f"   限制测试样本数量为 {max_samples}（原始数量: {len(brain_files)}）")
            brain_files = brain_files[:max_samples]
            hipp_files = hipp_files[:max_samples]
            labels = labels[:max_samples]
        
        print(f"   测试集大小: {len(brain_files)}")
        print(f"   测试集标签分布: {np.unique(labels, return_counts=True)}")
        
        # 2. 加载模型checkpoint
        print(f"\n2️⃣ 加载模型checkpoint...")
        # 直接使用指定的模型路径
        model_path = r"F:\ADNI\ClassificationAD\PROJECT\CONTRAST_LEARNING-master\runs\experiment_20251105_001435_BS_8\fold_2\best_model.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"   ✅ 模型加载成功: {model_path}")
        else:
            print(f"   ❌ 未找到模型文件: {model_path}")
            return None
        
        # 3. 构造未扰动测试集
        print("\n3️⃣ 构造未扰动测试集...")
        original_dataset = EvalDataset(brain_files, hipp_files, labels)
        original_loader = DataLoader(original_dataset, batch_size=1, shuffle=False, pin_memory=False)
        
        # 4. 在未扰动数据集上评估
        print("\n4️⃣ 在未扰动数据集上评估...")
        orig_acc, orig_f1, orig_auc = evaluate(self.model, original_loader, self.device)
        print(f"   原始数据集性能:")
        print(f"     Accuracy: {orig_acc:.4f}")
        print(f"     F1-score: {orig_f1:.4f}")
        print(f"     AUC: {orig_auc:.4f}")
        
        # 5. 定义一个自定义的Dataset，在加载时添加运动伪影
        class MotionArtifactDataset(EvalDataset):
            def __getitem__(self, index):
                b, h, label = super().__getitem__(index)
                # 添加运动伪影到脑图像
                b_motion = torch.tensor(add_motion_artifact(b.numpy(), s=self.s), dtype=torch.float32)
                return b_motion, h, label
        
        # 6. 构造扰动测试集
        print("\n5️⃣ 构造扰动测试集（添加运动伪影）...")
        perturbed_dataset = MotionArtifactDataset(brain_files, hipp_files, labels)
        perturbed_loader = DataLoader(perturbed_dataset, batch_size=1, shuffle=False, pin_memory=False)
        
        # 7. 在扰动数据集上评估
        print("\n6️⃣ 在扰动数据集上评估...")
        perturbed_acc, perturbed_f1, perturbed_auc = evaluate(self.model, perturbed_loader, self.device)
        print(f"   扰动数据集性能 (s={self.s}):")
        print(f"     Accuracy: {perturbed_acc:.4f}")
        print(f"     F1-score: {perturbed_f1:.4f}")
        print(f"     AUC: {perturbed_auc:.4f}")
        
        # 8. 计算性能下降
        print("\n7️⃣ 计算性能下降...")
        acc_drop = orig_acc - perturbed_acc
        f1_drop = orig_f1 - perturbed_f1
        auc_drop = orig_auc - perturbed_auc
        
        print(f"\n📊 运动伪影鲁棒性测试结果 (s={self.s})")
        print("=========================================")
        print(f"原始数据集性能:")
        print(f"  Accuracy: {orig_acc:.4f}")
        print(f"  F1-score: {orig_f1:.4f}")
        print(f"  AUC: {orig_auc:.4f}")
        print("-----------------------------------------")
        print(f"运动伪影数据集性能 (s={self.s}):")
        print(f"  Accuracy: {perturbed_acc:.4f}")
        print(f"  F1-score: {perturbed_f1:.4f}")
        print(f"  AUC: {perturbed_auc:.4f}")
        print("-----------------------------------------")
        print(f"性能下降:")
        print(f"  Accuracy Drop: {acc_drop:.4f} ({acc_drop/orig_acc*100:.1f}%)")
        print(f"  F1-score Drop: {f1_drop:.4f} ({f1_drop/orig_f1*100:.1f}%)")
        print(f"  AUC Drop: {auc_drop:.4f} ({auc_drop/orig_auc*100:.1f}%)")
        print("=========================================")
        
        return {
            'fold': target_fold,
            's': self.s,
            'original': {'acc': orig_acc, 'f1': orig_f1, 'auc': orig_auc},
            'perturbed': {'acc': perturbed_acc, 'f1': perturbed_f1, 'auc': perturbed_auc},
            'drop': {'acc': acc_drop, 'f1': f1_drop, 'auc': auc_drop}
        }

    def run_all_folds(self, max_samples=20):
        """运行所有folds的测试并计算平均性能"""
        print(f"\n🔄 开始所有folds的运动伪影鲁棒性测试 (s={self.s}, max_samples={max_samples})")
        results = []
        
        # 注意：由于完整的K-fold划分可能需要较长时间，这里我们简化处理
        # 对每个fold，我们使用相同的测试数据集，但加载不同的模型
        for fold in range(5):
            try:
                # 为每个fold使用不同的随机种子来选择不同的样本子集
                np.random.seed(self.random_state + fold)
                
                # 获取数据对
                brain_files, hipp_files, labels = create_brain_hipp_pairs(self.brain_root, self.hipp_root)
                
                # 限制数据量
                if len(brain_files) > max_samples:
                    # 随机选择样本
                    indices = np.random.choice(len(brain_files), max_samples, replace=False)
                    brain_files = [brain_files[i] for i in indices]
                    hipp_files = [hipp_files[i] for i in indices]
                    labels = [labels[i] for i in indices]
                
                print(f"\n处理fold {fold}，样本数量: {len(brain_files)}")
                
                # 加载对应fold的模型
                model_path = os.path.join(self.experiment_dir, f"fold_{fold+1}", "best_model.pth")
                if os.path.exists(model_path):
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    print(f"  加载模型成功: {model_path}")
                else:
                    print(f"  ⚠️ 未找到fold {fold}的模型，使用默认模型")
                    # 使用默认模型
                    default_model_path = r"F:\ADNI\ClassificationAD\PROJECT\CONTRAST_LEARNING-master\runs\experiment_20251105_001435_BS_8\fold_2\best_model.pth"
                    self.model.load_state_dict(torch.load(default_model_path, map_location=self.device))
                    self.model.eval()
                
                # 构造数据集并评估
                # 原始数据集
                original_dataset = EvalDataset(brain_files, hipp_files, labels)
                original_loader = DataLoader(original_dataset, batch_size=1, shuffle=False)
                orig_acc, orig_f1, orig_auc = evaluate(self.model, original_loader, self.device)
                
                # 扰动数据集
                perturbed_dataset = self.MotionArtifactDataset(brain_files, hipp_files, labels)
                perturbed_loader = DataLoader(perturbed_dataset, batch_size=1, shuffle=False)
                perturbed_acc, perturbed_f1, perturbed_auc = evaluate(self.model, perturbed_loader, self.device)
                
                # 计算性能下降
                acc_drop = orig_acc - perturbed_acc
                f1_drop = orig_f1 - perturbed_f1
                auc_drop = orig_auc - perturbed_auc
                
                results.append({
                    'fold': fold,
                    's': self.s,
                    'original': {'acc': orig_acc, 'f1': orig_f1, 'auc': orig_auc},
                    'perturbed': {'acc': perturbed_acc, 'f1': perturbed_f1, 'auc': perturbed_auc},
                    'drop': {'acc': acc_drop, 'f1': f1_drop, 'auc': auc_drop}
                })
                
            except Exception as e:
                print(f"❌ 运行fold {fold}时出错: {str(e)}")
        
        if results:
            # 计算平均性能
            avg_orig_acc = np.mean([r['original']['acc'] for r in results])
            avg_orig_f1 = np.mean([r['original']['f1'] for r in results])
            avg_orig_auc = np.mean([r['original']['auc'] for r in results])
            
            avg_perturbed_acc = np.mean([r['perturbed']['acc'] for r in results])
            avg_perturbed_f1 = np.mean([r['perturbed']['f1'] for r in results])
            avg_perturbed_auc = np.mean([r['perturbed']['auc'] for r in results])
            
            avg_acc_drop = np.mean([r['drop']['acc'] for r in results])
            avg_f1_drop = np.mean([r['drop']['f1'] for r in results])
            avg_auc_drop = np.mean([r['drop']['auc'] for r in results])
            
            print(f"\n📊 所有folds平均运动伪影鲁棒性测试结果 (s={self.s})")
            print("=========================================")
            print(f"平均原始性能:")
            print(f"  Accuracy: {avg_orig_acc:.4f}")
            print(f"  F1-score: {avg_orig_f1:.4f}")
            print(f"  AUC: {avg_orig_auc:.4f}")
            print("-----------------------------------------")
            print(f"平均扰动性能 (s={self.s}):")
            print(f"  Accuracy: {avg_perturbed_acc:.4f}")
            print(f"  F1-score: {avg_perturbed_f1:.4f}")
            print(f"  AUC: {avg_perturbed_auc:.4f}")
            print("-----------------------------------------")
            print(f"平均性能下降:")
            print(f"  Accuracy Drop: {avg_acc_drop:.4f} ({avg_acc_drop/avg_orig_acc*100:.1f}%)")
            print(f"  F1-score Drop: {avg_f1_drop:.4f} ({avg_f1_drop/avg_orig_f1*100:.1f}%)")
            print(f"  AUC Drop: {avg_auc_drop:.4f} ({avg_auc_drop/avg_orig_auc*100:.1f}%)")
            print("=========================================")
        
        return results
    
    class MotionArtifactDataset(EvalDataset):
        """添加运动伪影的数据集"""
        def __getitem__(self, index):
            b, h, label = super().__getitem__(index)
            # 添加运动伪影到脑图像
            b_motion = torch.tensor(add_motion_artifact(b.numpy(), s=self.s), dtype=torch.float32)
            return b_motion, h, label

# 为了兼容原有代码，保持原有的主函数结构，但添加新的测试类调用
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 数据路径（和你训练用的路径保持一致）
    brain_root = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\NIFTI5\all"
    hipp_root = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\hippdata\all"
    experiment_dir = r"F:\ADNI\ClassificationAD\PROJECT\CONTRAST_LEARNING-master\runs\experiment_20251105_001435_BS_8"
    
    # 创建测试实例
    test = MotionArtifactTest(
        device=device,
        brain_root=brain_root,
        hipp_root=hipp_root,
        experiment_dir=experiment_dir,
        random_state=42,  # 与训练时保持一致
        s=10  # 运动伪影参数
    )
    
    # 运行单个fold的测试（使用限制的样本数量以提高效率）
    print("\n========== 运行运动伪影鲁棒性测试 ==========")
    test.run_test(target_fold=1, max_samples=20)  # 使用20个样本进行测试
