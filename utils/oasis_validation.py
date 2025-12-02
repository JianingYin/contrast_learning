import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                           roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score, \
                           classification_report
from datasets.datasets_5fold import load_oasis_independent_validation, EvalDataset
from utils.evaluation_metrics import compute_auc
from utils.painter import plot_roc, plot_precision_recall, plot_confusion_matrix


def run_oasis_validation(best_model_class, experiment_dir, folds, config):
    """
    运行OASIS外部验证
    
    Args:
        best_model_class: 模型类，用于创建模型实例
        experiment_dir: 实验目录路径
        folds: 交叉验证折的数据
        config: 配置参数字典
    """


    print("\n" + "="*60)
    print("开始阶段2：OASIS外部验证")
    print("="*60)
    
    # 找到验证集上AUC最高的最佳折
    fold_aucs = []
    for fold_idx, fold in enumerate(folds):
        fold_dir = os.path.join(experiment_dir, f"fold_{fold_idx+1}")
        # 加载该折的最佳模型
        model = best_model_class().to(config['device'])
        model_path = os.path.join(fold_dir, "best_model.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告：折 {fold_idx+1} 的最佳模型文件不存在: {model_path}")
            fold_aucs.append(0.0)
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=config['device']))
        model.eval()
        
        # 准备该折的验证数据
        val_b, val_h, val_y = fold['val_brain_files'], fold['val_hipp_files'], fold['val_labels']
        val_dataset = EvalDataset(val_b, val_h, val_y)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
        
        # 在验证集上计算AUC
        val_labels, val_probs = [], []
        with torch.no_grad():
            for (b, h), labels in val_loader:
                b, h, labels = map(lambda x: x.to(config['device']), [b, h, labels])
                mid_feat = model.backbone(b, h)
                feat = model.encoder(mid_feat)
                logits = model.cls_head(feat)
                probs = torch.softmax(logits, dim=1)
                
                val_labels.extend(labels.cpu().numpy())
                val_probs.append(probs.cpu().numpy())
        
        # 处理概率数组
        val_probs = np.concatenate(val_probs) if val_probs else np.array([])
        val_labels = np.array(val_labels)
        
        # 计算AUC
        fold_auc = compute_auc(val_labels, val_probs, f"折{fold_idx+1}验证")
        fold_aucs.append(fold_auc)
        print(f"折 {fold_idx+1} 验证集 AUC: {fold_auc:.4f}")
    
    # 找到AUC最高的折
    best_fold_idx = np.argmax(fold_aucs)
    best_fold_auc = fold_aucs[best_fold_idx]
    print(f"\n最佳模型：折 {best_fold_idx+1}，验证集 AUC = {best_fold_auc:.4f}")
    
    # OASIS数据集路径 - 使用os.path.join确保跨平台兼容性
    # 注意：根据实际运行环境调整这些路径
    try:
        # 尝试使用相对路径或配置中的路径
        OAS_brain = os.path.join(".", "datasets", "OASIS", "OAS_brain_resampled")
        OAS_hippo = os.path.join(".", "datasets", "OASIS", "OAS_hippo", "Lside_resampled")
        
        # 如果上述路径不存在，尝试Windows和Linux常见路径
        if not os.path.exists(OAS_brain):
            # Windows路径
            OAS_brain = "F:/ADNI/ADNI_PNG_3Ddata/download_data/NIFTI_data/OAS_brain_resampled"
            OAS_hippo = "F:/ADNI/ADNI_PNG_3Ddata/download_data/NIFTI_data/OAS_hippo/Lside_resampled"
            
        # 最终尝试Linux路径
        if not os.path.exists(OAS_brain):
            OAS_brain = "/root/PROJECT/CONTRAST_LEARNING-master/datasets/OASIS/OAS_brain_mini"
            OAS_hippo = "/root/PROJECT/CONTRAST_LEARNING-master/datasets/OASIS/OAS_hippo_mini"
            
        print(f"使用OASIS数据集路径 - brain: {OAS_brain}")
        print(f"使用OASIS数据集路径 - hippo: {OAS_hippo}")
    except Exception as e:
        print(f"路径设置错误: {e}")
        # 设置默认路径
        OAS_brain = "datasets/OASIS/OAS_brain_resampled"
        OAS_hippo = "datasets/OASIS/OAS_hippo/Lside_resampled"
    
    # 加载OASIS数据
    oasis_brain_files, oasis_hipp_files, oasis_labels = load_oasis_independent_validation(OAS_brain, OAS_hippo)
    
    # 创建OASIS数据集和数据加载器
    oasis_dataset = EvalDataset(oasis_brain_files, oasis_hipp_files, oasis_labels)
    print(f"OASIS 测试样本数: {len(oasis_dataset)}")
    oasis_loader = DataLoader(oasis_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # 加载最佳模型
    best_model = best_model_class().to(config['device'])
    best_model_path = os.path.join(experiment_dir, f"fold_{best_fold_idx+1}", "best_model.pth")
    best_model.load_state_dict(torch.load(best_model_path, map_location=config['device']))
    best_model.eval()
    
    # 在OASIS数据集上进行预测
    print("\n在OASIS数据集上进行预测...")
    oasis_preds, oasis_true_labels, oasis_probs = [], [], []
    
    with torch.no_grad():
        for (b, h), labels in tqdm(oasis_loader, desc="OASIS预测"):
            b, h, labels = map(lambda x: x.to(config['device']), [b, h, labels])
            mid_feat = best_model.backbone(b, h)
            feat = best_model.encoder(mid_feat)
            logits = best_model.cls_head(feat)
            # 打印模型输出形状以调试
            print(f"logits.shape = {logits.shape}")
            probs = torch.softmax(logits, dim=1)
            print(f"probs.shape = {probs.shape}")
            preds = torch.argmax(probs, dim=1)
            
            oasis_preds.extend(preds.cpu().numpy())
            oasis_true_labels.extend(labels.cpu().numpy())
            oasis_probs.append(probs.cpu().numpy())
    
    # 打印预测结果统计信息以调试
    print(f"预测结果统计 - preds数量: {len(oasis_preds)}, labels数量: {len(oasis_true_labels)}, probs批次: {len(oasis_probs) if isinstance(oasis_probs, list) else '已合并'}")
    
    # 处理结果数组
    oasis_probs = np.concatenate(oasis_probs) if oasis_probs else np.array([])
    oasis_preds = np.array(oasis_preds)
    oasis_true_labels = np.array(oasis_true_labels)
    
    # 打印合并后的数组形状
    print(f"合并后数组形状 - probs: {oasis_probs.shape if len(oasis_probs.shape) > 0 else '空数组'}")
    print(f"合并后数组形状 - preds: {oasis_preds.shape}")
    print(f"合并后数组形状 - labels: {oasis_true_labels.shape}")
    
    # 创建OASIS结果保存目录
    oasis_result_dir = os.path.join(experiment_dir, "oasis_validation")
    os.makedirs(oasis_result_dir, exist_ok=True)
    oasis_plot_dir = os.path.join(oasis_result_dir, "plots")
    os.makedirs(oasis_plot_dir, exist_ok=True)
    
    # 计算评估指标
    # 准确率
    accuracy = accuracy_score(oasis_true_labels, oasis_preds)
    
    # 精确率、召回率、F1值（加权平均）
    precision = precision_score(oasis_true_labels, oasis_preds, average='weighted', zero_division=0)
    recall = recall_score(oasis_true_labels, oasis_preds, average='weighted', zero_division=0)
    f1 = f1_score(oasis_true_labels, oasis_preds, average='weighted', zero_division=0)
    
    # 每个类别的精确率、召回率、F1值
    precision_per_class = precision_score(oasis_true_labels, oasis_preds, average=None, zero_division=0)
    recall_per_class = recall_score(oasis_true_labels, oasis_preds, average=None, zero_division=0)
    f1_per_class = f1_score(oasis_true_labels, oasis_preds, average=None, zero_division=0)
    
    # AUC（多分类，OvR）
    auc_score = compute_auc(oasis_true_labels, oasis_probs, "OASIS外部验证")
    
    # PR-AUC（多分类，每个类别单独计算然后加权平均）
    pr_auc_scores = []
    for i in range(config['num_classes']):
        try:
            # 检查oasis_probs形状是否匹配
            if len(oasis_probs.shape) > 1 and oasis_probs.shape[1] > i:
                # 创建二分类标签
                binary_labels = (oasis_true_labels == i).astype(int)
                if len(np.unique(binary_labels)) > 1:  # 确保有正样本和负样本
                    ap = average_precision_score(binary_labels, oasis_probs[:, i])
                    pr_auc_scores.append(ap)
                else:
                    pr_auc_scores.append(np.nan)
            else:
                print(f"警告：oasis_probs形状不匹配，类别{i}缺失，跳过PR-AUC")
                pr_auc_scores.append(np.nan)
        except Exception as e:
            print(f"类别 {i} PR-AUC计算错误: {e}")
            pr_auc_scores.append(np.nan)
    
    # 排除无效项计算PR-AUC
    valid_pr_auc = [x for x in pr_auc_scores if not np.isnan(x)]
    pr_auc = np.mean(valid_pr_auc) if valid_pr_auc else np.nan
    
    # MCC和Cohen's Kappa
    mcc = matthews_corrcoef(oasis_true_labels, oasis_preds)
    cohen_kappa = cohen_kappa_score(oasis_true_labels, oasis_preds)
    
    # 打印OASIS验证结果
    print("\n" + "="*60)
    print("OASIS外部验证结果")
    print("="*60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"加权精确率 (Precision): {precision:.4f}")
    print(f"加权召回率 (Recall): {recall:.4f}")
    print(f"加权F1值 (F1): {f1:.4f}")
    # 处理可能的nan值
    auc_str = f"{auc_score:.4f}" if not np.isnan(auc_score) else "NaN"
    pr_auc_str = f"{pr_auc:.4f}" if not np.isnan(pr_auc) else "NaN"
    print(f"AUC (OvR, weighted): {auc_str}")
    print(f"PR-AUC (平均): {pr_auc_str}")
    print(f"MCC: {mcc:.4f}")
    print(f"Cohen's Kappa: {cohen_kappa:.4f}")
    
    print("\n每个类别的指标:")
    class_names = ["AD", "CN", "MCI"]
    for i in range(config['num_classes']):
        print(f"类别 {class_names[i]}:")
        print(f"  精确率: {precision_per_class[i]:.4f}")
        print(f"  召回率: {recall_per_class[i]:.4f}")
        print(f"  F1值: {f1_per_class[i]:.4f}")
        if i < len(pr_auc_scores):
            # 处理可能的nan值
            pr_auc_val = pr_auc_scores[i]
            pr_auc_str = f"{pr_auc_val:.4f}" if not np.isnan(pr_auc_val) else "NaN"
            print(f"  PR-AUC: {pr_auc_str}")
    
    # 生成分类报告
    print("\nOASIS分类报告:")
    print(classification_report(oasis_true_labels, oasis_preds, target_names=class_names, digits=4))
    
    # 保存结果到文件
    with open(os.path.join(oasis_result_dir, "oasis_results.txt"), 'w') as f:
        f.write("OASIS外部验证结果\n")
        f.write(f"最佳模型: 折 {best_fold_idx+1}，验证集 AUC = {best_fold_auc:.4f}\n\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"加权精确率 (Precision): {precision:.4f}\n")
        f.write(f"加权召回率 (Recall): {recall:.4f}\n")
        f.write(f"加权F1值 (F1): {f1:.4f}\n")
        # 处理可能的nan值
        auc_str = f"{auc_score:.4f}" if not np.isnan(auc_score) else "NaN"
        pr_auc_str = f"{pr_auc:.4f}" if not np.isnan(pr_auc) else "NaN"
        f.write(f"AUC (OvR, weighted): {auc_str}\n")
        f.write(f"PR-AUC (平均): {pr_auc_str}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Cohen's Kappa: {cohen_kappa:.4f}\n\n")
        
        f.write("每个类别的指标:\n")
        for i in range(config['num_classes']):
            f.write(f"类别 {class_names[i]}:\n")
            f.write(f"  精确率: {precision_per_class[i]:.4f}\n")
            f.write(f"  召回率: {recall_per_class[i]:.4f}\n")
            f.write(f"  F1值: {f1_per_class[i]:.4f}\n")
            if i < len(pr_auc_scores):
                # 处理可能的nan值
                pr_auc_val = pr_auc_scores[i]
                pr_auc_str = f"{pr_auc_val:.4f}" if not np.isnan(pr_auc_val) else "NaN"
                f.write(f"  PR-AUC: {pr_auc_str}\n")
        
        f.write("\n分类报告:\n")
        f.write(classification_report(oasis_true_labels, oasis_preds, target_names=class_names, digits=4))
    
    # 绘制ROC曲线
    plot_roc(oasis_true_labels, oasis_probs, config['num_classes'], 0, oasis_plot_dir)  # fold=0表示外部验证
    
    # 绘制PR曲线
    plot_precision_recall(oasis_true_labels, oasis_probs, config['num_classes'], 0, oasis_plot_dir)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(oasis_true_labels, oasis_preds, 0, oasis_plot_dir, class_names=class_names)
    
    print("\nOASIS外部验证完成！")
    print(f"结果保存至: {oasis_result_dir}")
    print(f"图表保存至: {oasis_plot_dir}")
    
    # 返回验证结果，便于后续使用
    return {
        'best_fold_idx': best_fold_idx,
        'best_fold_auc': best_fold_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'cohen_kappa': cohen_kappa,
        'oasis_result_dir': oasis_result_dir
    }