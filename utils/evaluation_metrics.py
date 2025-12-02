import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

def compute_auc(labels, probs, phase='train'):
    """计算多分类AUC（OvR + weighted），自动防止单类/维度错误，静默处理异常"""
    try:
        labels = np.array(labels)
        probs = np.array(probs, dtype=float)

        # 检查输入形状
        if probs.ndim != 2 or probs.shape[0] != len(labels):
            return {"roc_auc_ovr_macro": np.nan, "roc_auc_ovr_weighted": np.nan}

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            return {"roc_auc_ovr_macro": np.nan, "roc_auc_ovr_weighted": np.nan}  # 用 nan ，不影响均值计算

        auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        return auc

        # OvR 形式的多分类 ROC-AUC
        roc_auc_ovr_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        roc_auc_ovr_weighted = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')

        return {
            "roc_auc_ovr_macro": roc_auc_ovr_macro,
            "roc_auc_ovr_weighted": roc_auc_ovr_weighted
        }

    except Exception:
        return {"roc_auc_ovr_macro": np.nan, "roc_auc_ovr_weighted": np.nan}

def compute_pr_auc(labels, probs):
    """计算多分类 PR-AUC（OvR平均值），自动防止单类错误"""
    try:
        labels = np.array(labels)
        probs = np.array(probs, dtype=float)

        if probs.ndim != 2 or probs.shape[0] != len(labels):
            return np.nan

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            return np.nan

        labels_bin = label_binarize(labels, classes=unique_classes)
        pr_auc_mean = average_precision_score(labels_bin, probs, average='macro')
        return pr_auc_mean

    except Exception:
        return np.nan


def epoch_output_params(train_correct, train_total, val_correct, val_total,
                        train_preds, train_labels, val_preds, val_labels,
                        train_probs, val_probs, train_loss, val_loss, epoch):
    """计算每轮训练和验证的各项指标"""
    
    # 准确率
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    # 精确率、召回率、F1
    train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
    val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
    train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
    train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

    # AUC（多分类）
    train_auc_dict = compute_auc(train_labels, train_probs)
    val_auc_dict = compute_auc(val_labels, val_probs)

    # === PR-AUC（平均值） ===
    train_pr_auc_mean = compute_pr_auc(train_labels, train_probs)
    val_pr_auc_mean = compute_pr_auc(val_labels, val_probs)

    # 安全地处理可能为nan的值
    def safe_round(value, decimals=4):
        return round(value, decimals) if not isinstance(value, float) or not np.isnan(value) else value
    
    # 返回结果字典
    return {
        "epoch": epoch + 1,
        "train_loss": safe_round(train_loss),
        "val_loss": safe_round(val_loss),
        "train_acc": safe_round(train_acc),
        "val_acc": safe_round(val_acc),
        "train_precision": safe_round(train_precision),
        "val_precision": safe_round(val_precision),
        "train_recall": safe_round(train_recall),
        "val_recall": safe_round(val_recall),
        "train_f1": safe_round(train_f1),
        "val_f1": safe_round(val_f1),
    
        # === ROC-AUC ===
        "train_roc_auc_ovr_macro": safe_round(train_auc_dict["roc_auc_ovr_macro"]),
        "val_roc_auc_ovr_macro": safe_round(val_auc_dict["roc_auc_ovr_macro"]),
        "train_roc_auc_ovr_weighted": safe_round(train_auc_dict["roc_auc_ovr_weighted"]),
        "val_roc_auc_ovr_weighted": safe_round(val_auc_dict["roc_auc_ovr_weighted"]),
    
        # === 为兼容旧字段，AUC = ROC-AUC(weighted) ===
        "train_auc": safe_round(train_auc_dict["roc_auc_ovr_weighted"]),
        "val_auc": safe_round(val_auc_dict["roc_auc_ovr_weighted"]),
    
        # === PR-AUC ===
        "train_pr_auc_mean": safe_round(train_pr_auc_mean),
        "val_pr_auc_mean": safe_round(val_pr_auc_mean)
    }

