        
import os, glob, csv, sys
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

# 自定义模块导入
from datasets.datasets_class import PairedContrastiveDataset, EvalDataset
from utils.supcon_loss import SupConLoss
from utils.StratifiedKFold import get_stratified_kfold_lists
from utils.painter import (
    plot_metrics,
    plot_confusion_matrix,
    plot_roc,
    plot_precision_recall,
    plot_metrics_distribution_boxplot,
    plot_metrics_distribution_barchart
)
from utils.oasis_validation import run_oasis_validation
from utils.evaluation_metrics import *
from utils.metrics_kappa_mcc import add_metrics_to_dict
from models.model.modelV24 import FullModel
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ================================
# 早停机制
# ================================
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ================================
# 创建实验目录
# ================================
def create_experiment_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("./runs", f"experiment_{timestamp}")
    plot_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return experiment_dir


# ================================
# 训练单折
# ================================
def train_fold(fold_idx, train_b, train_h, train_y, val_b, val_h, val_y, experiment_dir, config):
    device = config['device']
    batch_size = config['batch_size']
    epochs = config['epochs']
    supcon_weight = config['supcon_weight']
    lr = config['lr']
    num_classes = config['num_classes']

    print(f"\n{'=' * 50}\n开始训练第 {fold_idx + 1}/{config['n_folds']} 折\n{'=' * 50}")

    # 目录
    fold_dir = os.path.join(experiment_dir, f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    
    plot_dir = os.path.join(fold_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fold_log_file = os.path.join(fold_dir, "train_log.csv")

    with open(fold_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_acc", "val_loss", "val_acc",
            "train_precision", "val_precision",
            "train_recall", "val_recall",
            "train_f1", "val_f1",
            "train_auc", "val_auc",
            "train_kappa", "val_kappa",
            "train_mcc", "val_mcc"
        ])




    # 数据
    train_dataset = PairedContrastiveDataset(train_b, train_h, train_y)
    val_dataset = EvalDataset(val_b, val_h, val_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FullModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    contrastive_criterion = SupConLoss(temperature=0.07).to(device)
    classification_criterion = nn.CrossEntropyLoss().to(device)
    early_stopping = EarlyStopping(patience=20, delta=0.0, path=os.path.join(fold_dir, "best_model.pt"))

    best_val_acc = 0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    # ================================
    # 训练循环
    # ================================
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        train_preds, train_labels, train_probs = [], [], []

        for (b1, h1), (b2, h2), labels in tqdm(train_loader, desc=f"[Fold {fold_idx + 1}] Epoch {epoch + 1}", leave=False):
            b1, h1, b2, h2, labels = map(lambda x: x.to(device), [b1, h1, b2, h2, labels])
            optimizer.zero_grad()

            mid_feat1 = model.backbone(b1, h1)
            mid_feat2 = model.backbone(b2, h2)
            feat1 = model.encoder(mid_feat1)
            feat2 = model.encoder(mid_feat2)
            proj1, proj2 = model.proj_head(feat1), model.proj_head(feat2)
            logits = model.cls_head(feat1)

            features = torch.cat([proj1, proj2], dim=0)
            contrastive_labels = torch.cat([labels, labels], dim=0)

            loss_contrast = contrastive_criterion(features, contrastive_labels)
            loss_class = classification_criterion(logits, labels)
            hybrid_loss = supcon_weight * loss_contrast + (1 - supcon_weight) * loss_class

            hybrid_loss.backward()
            optimizer.step()

            total_loss += hybrid_loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for (b, h), labels in val_loader:
                b, h, labels = map(lambda x: x.to(device), [b, h, labels])
                logits = model.cls_head(model.encoder(model.backbone(b, h)))
                loss = classification_criterion(logits, labels)
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[Fold {fold_idx + 1}] Early stopping.")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))

                # =========================
        # 计算每个epoch的训练/验证指标
        # =========================
        train_probs_np = np.concatenate(train_probs)
        val_probs_np = np.concatenate(all_probs)

        # 训练集指标
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='macro')
        train_auc = roc_auc_score(train_labels, train_probs_np, multi_class='ovr', average='macro')
        train_kappa = cohen_kappa_score(train_labels, train_preds)
        train_mcc = matthews_corrcoef(train_labels, train_preds)

        # 验证集指标
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        val_auc = roc_auc_score(all_labels, val_probs_np, multi_class='ovr', average='macro')
        val_kappa = cohen_kappa_score(all_labels, all_preds)
        val_mcc = matthews_corrcoef(all_labels, all_preds)

        # 写入CSV
        with open(fold_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss, train_acc, val_loss, val_acc,
                train_precision, val_precision,
                train_recall, val_recall,
                train_f1, val_f1,
                train_auc, val_auc,
                train_kappa, val_kappa,
                train_mcc, val_mcc
            ])
        tqdm.write(f"Epoch {epoch + 1}: TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")


    # 绘图
    plot_metrics(train_accs, val_accs, train_losses, val_losses, fold=fold_idx + 1, save_dir=plot_dir)
    plot_confusion_matrix(all_labels, all_preds, fold=fold_idx + 1, save_dir=plot_dir, class_names=["AD", "CN", "MCI"])
    plot_roc(all_labels, np.concatenate(all_probs), num_classes=num_classes, fold=fold_idx + 1, save_dir=plot_dir)
    plot_precision_recall(all_labels, np.concatenate(all_probs), num_classes=num_classes, fold=fold_idx + 1, save_dir=plot_dir)

    return {
        'best_val_acc': best_val_acc,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': np.concatenate(all_probs)
    }


# ================================
# 主程序入口
# ================================
if __name__ == "__main__":
    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 50,
        'supcon_weight': 0.1,
        'num_classes': 3,
        'n_folds': 5,
        'random_state': 42
    }

    brain_root = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\NIFTI5\all"
    hipp_root = r"F:\ADNI\ADNI_PNG_3Ddata\download_data\NIFTI_data\hippdata\all"
    folds = get_stratified_kfold_lists(brain_root, hipp_root, n_splits=config['n_folds'], random_state=config['random_state'])

    experiment_dir = create_experiment_dir()
    with open(os.path.join(experiment_dir, "config.txt"), 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    cv_results = []
    all_predictions, all_ground_truth, all_probabilities = [], [], []

    for fold_idx, fold in enumerate(folds):
        train_y = [int(y) for y in fold['train_labels']]
        val_y = [int(y) for y in fold['val_labels']]
        result = train_fold(fold_idx, fold['train_brain_files'], fold['train_hipp_files'], train_y,
                            fold['val_brain_files'], fold['val_hipp_files'], val_y, experiment_dir, config)
        cv_results.append(result)
        all_predictions.extend(result['all_preds'])
        all_ground_truth.extend(result['all_labels'])
        all_probabilities.extend(result['all_probs'])

    # 汇总结果
    avg_val_acc = np.mean([r['best_val_acc'] for r in cv_results])
    print(f"\n平均最佳验证准确率: {avg_val_acc:.4f}")

    # 保存汇总指标
    cv_metrics = {
        'accuracy': [accuracy_score(r['all_labels'], r['all_preds']) for r in cv_results],
        'f1': [precision_recall_fscore_support(r['all_labels'], r['all_preds'], average='macro')[2] for r in cv_results],
        'auc': [roc_auc_score(r['all_labels'], r['all_probs'], average='macro', multi_class='ovr') for r in cv_results]
    }

    # ================================
    # 保存箱线图
    # ================================
    plot_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print("\n保存5折指标分布图...")
    plot_metrics_distribution_boxplot(cv_metrics, plot_dir)
    plot_metrics_distribution_barchart(cv_metrics, plot_dir)
    print(f"✅ 箱线图和条形图已保存至: {plot_dir}")


    summary_file = os.path.join(experiment_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("五折交叉验证指标摘要\n")
        f.write("="*40 + "\n")
    
        for metric, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"{metric:<10}: {mean_val:.4f} ± {std_val:.4f}\n")
    
        # 置信区间（假设正态分布近似）
        f.write("\n95% 置信区间估计:\n")
        for metric, values in cv_metrics.items():
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            ci_low = mean - 1.96 * std / np.sqrt(len(values))
            ci_high = mean + 1.96 * std / np.sqrt(len(values))
            f.write(f"{metric:<10}: [{ci_low:.4f}, {ci_high:.4f}]\n")

        # ================================
        # Bootstrap 置信区间计算函数（总体）
        # ================================
    
        def bootstrap_ci(y_true, y_pred, y_prob, metric_func, n_bootstrap=1000, random_state=42):
            np.random.seed(random_state)
            stats = []
            n = len(y_true)
            for _ in range(n_bootstrap):
                idx = np.random.choice(np.arange(n), size=n, replace=True)
                try:
                    stat = metric_func(y_true[idx], y_pred[idx], y_prob[idx] if y_prob is not None else None)
                    stats.append(stat)
                except Exception:
                    continue
            stats = np.array(stats)
            return np.percentile(stats, [2.5, 97.5])
    
        # 定义指标计算函数
        def auc_metric(y_true, y_pred, y_prob):
            return roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    
        def pr_auc_metric(y_true, y_pred, y_prob):
            return average_precision_score(y_true, y_prob, average='macro')
    
        def acc_metric(y_true, y_pred, y_prob=None):
            return accuracy_score(y_true, y_pred)
    
        def f1_metric(y_true, y_pred, y_prob=None):
            return precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
    
        # ================================
        # 总体 Bootstrap 置信区间计算
        # ================================
        all_labels_np = np.array(all_ground_truth)
        all_preds_np = np.array(all_predictions)
        all_probs_np = np.vstack(all_probabilities)
        
        f.write("\n总体 Bootstrap 置信区间 (95%):\n")
        for name, func in {
            "Accuracy": acc_metric,
            "F1": f1_metric,
            "AUC": auc_metric,
            "PR-AUC": pr_auc_metric
        }.items():
            ci_low, ci_high = bootstrap_ci(all_labels_np, all_preds_np, all_probs_np, func, n_bootstrap=1000)
            f.write(f"{name:<10}: [{ci_low:.4f}, {ci_high:.4f}]\n")



    # ================================
    # OASIS 独立验证集
    # ================================
    oasis_dir = os.path.join(experiment_dir, "oasis_result")
    os.makedirs(oasis_dir, exist_ok=True)
    oasis_result_file = os.path.join(oasis_dir, "oasis_eval.txt")
    
    # 调用独立验证函数
    oasis_metrics = run_oasis_validation(FullModel, experiment_dir, folds, config)
    with open(oasis_result_file, 'w') as f:
        for k, v in oasis_metrics.items():
            f.write(f"{k}: {v}\n")
    
    print(f"✅ OASIS 独立验证集结果已保存至: {oasis_result_file}")






    
