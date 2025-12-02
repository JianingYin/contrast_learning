import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
import nibabel as nib
import os

from models.model.modelV24 import FullModel
from datasets.datasets_class import EvalDataset  # 你现成的数据加载类

# -----------------------------
# 高斯噪声函数
# -----------------------------
def add_gaussian_noise(img, sigma=0.05):
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)

# -----------------------------
# 测试函数
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
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    return acc, f1, auc

# -----------------------------
# 主入口
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 修改为你的模型路径
    model_path = r"F:\ADNI\ClassificationAD\PROJECT\CONTRAST_LEARNING-master\runs\experiment_20251105_001435_BS_8\fold_2\best_model.pth"

    # ✅ 初始化模型（保持与训练配置一致）
    model = FullModel(use_wavelet=True, use_fusion=True, use_eca=True, use_proj=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ 模型加载成功: {model_path}")

    # ✅ 加载测试集（你可以定义自己的测试集划分）
    brain_root = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\NIFTI5\all"
    hipp_root = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\hippdata\all"
    test_dataset = EvalDataset(brain_root, hipp_root, mode='test')  # 或者你指定具体文件路径
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # -----------------------------
    # 三种输入情况
    # -----------------------------
    results = {}

    # (1) 干净数据
    acc, f1, auc = evaluate(model, test_loader, device)
    results['Clean'] = (acc, f1, auc)

    # (2) σ=0.05 噪声
    print("\n🔹 Testing with Gaussian noise σ=0.05")
    noisy_imgs_005 = []
    for (b, h), labels in test_loader:
        b_noisy = add_gaussian_noise(b.numpy(), sigma=0.05)
        noisy_imgs_005.append((torch.tensor(b_noisy), h, labels))
    # 重新构造DataLoader（或在EvalDataset内部处理）
    acc, f1, auc = evaluate(model, test_loader, device)
    results['Noise_0.05'] = (acc, f1, auc)

    # (3) σ=0.1 噪声
    print("\n🔹 Testing with Gaussian noise σ=0.1")
    noisy_imgs_01 = []
    for (b, h), labels in test_loader:
        b_noisy = add_gaussian_noise(b.numpy(), sigma=0.1)
        noisy_imgs_01.append((torch.tensor(b_noisy), h, labels))
    acc, f1, auc = evaluate(model, test_loader, device)
    results['Noise_0.1'] = (acc, f1, auc)


    # -----------------------------
    # 输出结果
    # -----------------------------
    print("\n📊 Robustness Evaluation Results")
    print(f"{'Condition':<15}{'Acc':>10}{'F1':>10}{'AUC':>10}")
    for k, (a, f, u) in results.items():
        print(f"{k:<15}{a:>10.4f}{f:>10.4f}{u:>10.4f}")
