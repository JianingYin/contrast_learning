import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# ===============================
# 🔧 全局绘图参数设置（推荐论文级输出）
# ===============================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300
})


# ==========================================================
# 📈 1. 训练 & 验证过程曲线：Accuracy / Loss
# ==========================================================
def plot_metrics(train_acc, val_acc, train_loss, val_loss, fold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_acc) + 1)

    # 绘制准确率曲线
    plt.figure()
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title(f'Accuracy Curve (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'acc_curve_fold{fold}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # 绘制损失曲线
    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title(f'Loss Curve (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0.0, 1.0)
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_curve_fold{fold}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================================
# 🧩 2. 混淆矩阵（Confusion Matrix）
# ==========================================================
def plot_confusion_matrix(y_true, y_pred, fold, save_dir, class_names=None):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Fold {fold})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_fold{fold}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================================
# 🩺 3. ROC 曲线
# ==========================================================
def plot_roc(y_true, y_pred_prob, num_classes, fold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_true_onehot = label_binarize(y_true, classes=range(num_classes))
    
    try:
        # 先尝试直接拼接，确保每个数组都是二维的
        y_pred_prob = np.concatenate([np.atleast_2d(p) for p in y_pred_prob], axis=0)
    except ValueError as e:
        print(f"[ROC] 处理预测概率时出错: {e}")
        # 备选方案：找出最长概率向量并补零
        y_pred_prob_list = []
        max_len = max(np.array(p).size for p in y_pred_prob)  # 找出最长概率向量
        for p in y_pred_prob:
            p_array = np.array(p).flatten()
            # 如果长度不够，则补零
            if len(p_array) < max_len:
                p_array = np.pad(p_array, (0, max_len - len(p_array)), mode='constant', constant_values=0)
            y_pred_prob_list.append(p_array)
        y_pred_prob = np.vstack(y_pred_prob_list)
        print(f"[ROC] 自动修正预测概率维度不一致，统一到 {max_len} 列")
        num_classes = min(num_classes, y_pred_prob.shape[1])


    plt.figure(figsize=(7, 6))
    for i in range(num_classes):
        if i >= y_pred_prob.shape[1]:
            print(f"[ROC] Fold {fold}: 类别 {i} 概率列不存在，跳过")
            continue
        if np.sum(y_true_onehot[:, i]) == 0:
            print(f"[ROC] Fold {fold}: 类别 {i} 样本不足，跳过")
            continue
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f'ROC Curve (Fold {fold})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'roc_curve_fold{fold}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================================
# 📊 4. Precision-Recall 曲线
# ==========================================================
def plot_precision_recall(y_true, y_pred_prob, num_classes, fold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_true_onehot = label_binarize(y_true, classes=range(num_classes))
    
    try:
        # 先尝试直接拼接，确保每个数组都是二维的
        y_pred_prob = np.concatenate([np.atleast_2d(p) for p in y_pred_prob], axis=0)
    except ValueError as e:
        print(f"[PR] 处理预测概率时出错: {e}")
        # 备选方案：找出最长概率向量并补零
        y_pred_prob_list = []
        max_len = max(np.array(p).size for p in y_pred_prob)
        print(f"[PR] 自动修正预测概率维度不一致，统一到 {max_len} 列")
        for p in y_pred_prob:
            p_array = np.array(p).flatten()
            if len(p_array) < max_len:
                p_array = np.pad(p_array, (0, max_len - len(p_array)), mode='constant', constant_values=0)
            y_pred_prob_list.append(p_array)
        y_pred_prob = np.vstack(y_pred_prob_list)

    num_classes = min(num_classes, y_pred_prob.shape[1])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if i >= y_pred_prob.shape[1]:
            print(f"[PR] Fold {fold}: 类别 {i} 概率列不存在，跳过")
            continue
        if np.sum(y_true_onehot[:, i]) == 0:
            print(f"[PR] Fold {fold}: 类别 {i} 样本不足，跳过")
            continue

        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i], y_pred_prob[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_pred_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AP={ap:.2f})')

    # 添加参考线
    for i in range(num_classes):
        if np.sum(y_true_onehot[:, i]) > 0:
            rate = np.sum(y_true_onehot[:, i]) / len(y_true_onehot)
            plt.axhline(y=rate, linestyle='--', color='gray', alpha=0.5)

    plt.title(f'Precision-Recall Curve (Fold {fold})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'precision_recall_fold{fold}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================================
# 📦 5. 五折指标分布：箱线图
# ==========================================================
def plot_metrics_distribution_boxplot(metrics_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 自动选择可绘制指标
    metric_names = [m for m in metrics_dict.keys() if isinstance(metrics_dict[m], (list, np.ndarray))]
    data_to_plot = [metrics_dict[m] for m in metric_names]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, orient='v', palette='pastel')
    plt.boxplot(data_to_plot, patch_artist=True, showmeans=True, meanline=True)
    plt.xticks(ticks=np.arange(1, len(metric_names)+1), labels=metric_names, rotation=15)
    plt.title('5-Fold Cross Validation Metrics Distribution')
    plt.ylabel('Metric Value')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'metrics_distribution_boxplot.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================================
# 📊 6. 五折指标均值条形图
# ==========================================================
def plot_metrics_distribution_barchart(metrics_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'kappa', 'mcc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'Kappa', 'MCC']

    means = [np.mean(metrics_dict[m]) for m in metric_names if m in metrics_dict]
    stds = [np.std(metrics_dict[m]) for m in metric_names if m in metrics_dict]
    labels = [label for m, label in zip(metric_names, metric_labels) if m in metrics_dict]

    x = np.arange(len(labels))
    width = 0.6

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, means, width, yerr=stds, capsize=5, color='skyblue')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.title('5-Fold Cross Validation Metrics Mean Values')
    plt.ylabel('Value')
    plt.xticks(x, labels)
    plt.ylim([0.0, 1.1])
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution_barchart.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
