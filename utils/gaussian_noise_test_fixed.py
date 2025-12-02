# gaussian_noise_test_fixed.py
"""
严格版 Gaussian Noise 测试脚本 - 修复版
使用动态导入解决ModuleNotFoundError问题
"""

import os
import sys
import inspect
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import importlib.util

try:
    import nibabel as nib
except Exception:
    nib = None

# ====== 动态导入所有模块 ======
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"项目根目录: {project_root}")

# 1. 动态导入FullModel
model_path = os.path.join(project_root, 'models', 'model', 'modelV24.py')
if os.path.exists(model_path):
    print(f"找到模型文件: {model_path}")
    spec = importlib.util.spec_from_file_location("modelV24", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    FullModel = model_module.FullModel
    print("✓ 成功加载FullModel")
else:
    raise ImportError(f"无法找到模型文件: {model_path}")

# 2. 动态导入datasets_class模块
dataset_path = os.path.join(project_root, 'datasets', 'datasets_class.py')
if os.path.exists(dataset_path):
    print(f"找到数据集文件: {dataset_path}")
    spec = importlib.util.spec_from_file_location("datasets_class", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    collect_file_label_pairs = dataset_module.collect_file_label_pairs
    # 尝试加载EvalDataset
    try:
        EvalDataset = dataset_module.EvalDataset
        USE_EVALDATASET = True
        print("✓ 成功加载EvalDataset")
    except:
        EvalDataset = None
        USE_EVALDATASET = False
        print("⚠ EvalDataset不可用")
else:
    raise ImportError(f"无法找到数据集文件: {dataset_path}")

# 3. 动态导入StratifiedKFold模块
kfold_path = os.path.join(project_root, 'utils', 'StratifiedKFold.py')
if os.path.exists(kfold_path):
    print(f"找到K折划分文件: {kfold_path}")
    spec = importlib.util.spec_from_file_location("StratifiedKFold", kfold_path)
    kfold_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kfold_module)
    get_stratified_kfold_lists = kfold_module.get_stratified_kfold_lists
    print("✓ 成功加载get_stratified_kfold_lists")
else:
    raise ImportError(f"无法找到K折划分文件: {kfold_path}")

# ====== 参数 ======
BRAIN_ROOT = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\NIFTI5\all"
HIPP_ROOT  = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\hippdata\all"
EXPERIMENT_DIR = r"F:\ADNI\ClassificationAD\PROJECT\CONTRAST_LEARNING-master\runs\experiment_20251105_001435_BS_8"

N_SPLITS = 5
RANDOM_STATE = 42
SIGMA = 0.10              # Gaussian Noise 强度
PERTURB_HPP = False
BATCH_SIZE = 1
NUM_WORKERS = 0

# ====== 安全加载 checkpoint ======
def safe_load_checkpoint(model, ckpt_path, map_location=None):
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "net"):
            if key in state:
                try:
                    model.load_state_dict(state[key])
                    return
                except:
                    pass
        try:
            model.load_state_dict(state)
            return
        except:
            pass
    raise RuntimeError(f"无法加载 checkpoint：{ckpt_path}")

# ====== Gaussian Noise 核心函数 ======
def add_gaussian_noise(img, sigma=0.1):
    """
    img: numpy array 或 torch tensor
    返回同 shape 的 numpy float32
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    img = img.astype(np.float32)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)

    out = img + noise
    return out.astype(np.float32)

# ====== Dataset fallback loader ======
class SingleFileDataset(Dataset):
    def __init__(self, brain_files, hipp_files, labels, transform=None):
        if nib is None:
            raise RuntimeError("nibabel 未安装，无法加载 NIfTI 文件。")
        self.bf = brain_files
        self.hf = hipp_files
        self.y  = labels

    def __len__(self):
        return len(self.y)

    def _load_nifti(self, path):
        img = nib.load(path).get_fdata(dtype=np.float32)
        if img.ndim == 3:
            img = img[None]
        return torch.from_numpy(img.astype(np.float32))

    def __getitem__(self, idx):
        b = self._load_nifti(self.bf[idx])
        h = self._load_nifti(self.hf[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return b, h, y

class FirstInPairWrapper(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        (b1, h1), (b2, h2), y = item
        return b1, h1, torch.tensor(y, dtype=torch.long)

# ====== Gaussian Noise Wrapper ======
class GaussianNoiseWrapper(Dataset):
    def __init__(self, base_dataset, sigma=0.1, perturb_brain=True, perturb_hipp=False):
        self.base = base_dataset
        self.sigma = sigma
        self.pb = perturb_brain
        self.ph = perturb_hipp

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        # 兼容不同的返回格式
        item = self.base[idx]
        if len(item) == 3:
            b, h, y = item
        elif len(item) == 2 and isinstance(item[0], tuple):
            (b, h), y = item
        else:
            raise ValueError(f"不支持的数据格式: {item}")

        b = b if isinstance(b, torch.Tensor) else torch.tensor(b)
        h = h if isinstance(h, torch.Tensor) else torch.tensor(h)

        if self.pb:
            b_np = b.numpy()
            b = torch.from_numpy(add_gaussian_noise(b_np, sigma=self.sigma))
        if self.ph:
            h_np = h.numpy()
            h = torch.from_numpy(add_gaussian_noise(h_np, sigma=self.sigma))

        return b.float(), h.float(), torch.tensor(y, dtype=torch.long)

# ====== 评估函数 ======
def evaluate_model_on_loader(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=80):
            # 兼容不同的批处理格式
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    b, h, y = batch
                elif len(batch) == 2 and isinstance(batch[0], (list, tuple)):
                    (b, h), y = batch
                else:
                    raise ValueError(f"不支持的批处理格式: {batch}")
            else:
                raise TypeError(f"批处理应该是列表或元组，得到: {type(batch)}")

            b, h, y = b.to(device), h.to(device), y.to(device)

            out = model(b, h)
            logits = out[-1] if isinstance(out, (tuple, list)) else out
            probs = F.softmax(logits, dim=1)

            preds = probs.argmax(1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except:
        auc = np.nan

    return acc, f1, auc

# ====== 主类 ======
class GaussianNoiseTest:
    def __init__(self, device, brain_root, hipp_root, experiment_dir,
                 n_splits=5, random_state=42, sigma=0.1, perturb_hipp=False,
                 batch_size=1, num_workers=0):

        self.device = device
        self.brain_root = brain_root
        self.hipp_root  = hipp_root
        self.experiment_dir = experiment_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.sigma = sigma
        self.ph = perturb_hipp
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = FullModel().to(device)

    def _build_datasets(self, fold_idx):
        folds = get_stratified_kfold_lists(
            self.brain_root, self.hipp_root,
            n_splits=self.n_splits, random_state=self.random_state
        )
        info = folds[fold_idx]
        val_b = info["val_brain_files"]
        val_h = info["val_hipp_files"]
        val_y = info["val_labels"]

        # 优先尝试 EvalDataset
        base = None
        if USE_EVALDATASET:
            try:
                base = EvalDataset(val_b, val_h, val_y)
                sample = base[0]
                # 仅当 EvalDataset 返回三个元素，并且第一个元素是两输入才认为是 pair 数据
                if isinstance(sample, (list, tuple)):
                    if len(sample) == 3 and isinstance(sample[0], (list, tuple)) and len(sample[0]) == 2:
                        # ((b1,h1),(b2,h2),y)
                        base = FirstInPairWrapper(base)
            except:
                base = None

        if base is None:
            base = SingleFileDataset(val_b, val_h, val_y)

        return base, GaussianNoiseWrapper(base, sigma=self.sigma,
                                          perturb_brain=True, perturb_hipp=self.ph)

    def run_on_fold(self, fold_idx):
        print(f"\n=== Running Gaussian Noise Test fold {fold_idx} (sigma={self.sigma}) ===")

        ckpt = os.path.join(self.experiment_dir, f"fold_{fold_idx+1}", "best_model.pth")
        safe_load_checkpoint(self.model, ckpt, map_location=self.device)
        print("Loaded:", ckpt)

        orig_ds, noise_ds = self._build_datasets(fold_idx)

        orig_loader  = DataLoader(orig_ds, batch_size=self.batch_size, shuffle=False)
        noise_loader = DataLoader(noise_ds, batch_size=self.batch_size, shuffle=False)

        print("[INFO] Evaluating Original")
        acc0, f10, auc0 = evaluate_model_on_loader(self.model, orig_loader, self.device)
        print(f"Original   Acc={acc0:.4f}  F1={f10:.4f}  AUC={auc0:.4f}")

        print("[INFO] Evaluating Gaussian Noise")
        acc1, f11, auc1 = evaluate_model_on_loader(self.model, noise_loader, self.device)
        print(f"Noise σ={self.sigma}  Acc={acc1:.4f}  F1={f11:.4f}  AUC={auc1:.4f}")

        print(f"Drop: Acc={acc0 - acc1:.4f}, F1={f10 - f11:.4f}, AUC={auc0 - auc1:.4f}")
        return acc0, acc1

# ====== main ======
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    tester = GaussianNoiseTest(device,
                               BRAIN_ROOT, HIPP_ROOT, EXPERIMENT_DIR,
                               n_splits=N_SPLITS,
                               random_state=RANDOM_STATE,
                               sigma=SIGMA,
                               perturb_hipp=PERTURB_HPP,
                               batch_size=BATCH_SIZE)

    tester.run_on_fold(1)  # 测试第2折数据