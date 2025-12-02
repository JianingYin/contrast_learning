import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef


def calculate_kappa(y_true, y_pred, weights='quadratic'):
    """
    计算Cohen's Kappa系数
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        weights: 权重类型 ('linear', 'quadratic' 或 None)
        
    Returns:
        float: Cohen's Kappa系数值
    """
    try:
        kappa = cohen_kappa_score(y_true, y_pred, weights=weights)
        return kappa
    except Exception as e:
        print(f"计算Cohen's Kappa时出错: {e}")
        return np.nan


def calculate_mcc(y_true, y_pred):
    """
    计算Matthews相关系数 (MCC)
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        float: MCC值
    """
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        return mcc
    except Exception as e:
        print(f"计算MCC时出错: {e}")
        return np.nan


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_prob=None):
    """
    计算综合评估指标，包括额外的Kappa和MCC
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_prob: 预测概率 (可选)
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {}
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        print("⚠️ 输入长度不匹配，跳过Kappa/MCC计算")
        metrics.update({'kappa_quadratic': np.nan, 'kappa_linear': np.nan, 'kappa_none': np.nan, 'mcc': np.nan})
        return metrics
    
    # 计算Cohen's Kappa
    metrics['kappa_quadratic'] = calculate_kappa(y_true, y_pred, weights='quadratic')
    metrics['kappa_linear'] = calculate_kappa(y_true, y_pred, weights='linear')
    metrics['kappa_none'] = calculate_kappa(y_true, y_pred, weights=None)
    
    # 计算MCC
    metrics['mcc'] = calculate_mcc(y_true, y_pred)
    
    return metrics


def add_metrics_to_dict(base_metrics, y_true, y_pred):
    """
    将额外的指标添加到现有的metrics_dict中
    
    Args:
        base_metrics: 基础指标字典
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        dict: 更新后的指标字典
    """
    additional = calculate_comprehensive_metrics(y_true, y_pred)
    
    # 添加主要指标的简化版本
    base_metrics['kappa'] = additional['kappa_quadratic']  # 使用二次权重作为默认值
    base_metrics['mcc'] = additional['mcc']
    
    # 添加所有详细指标
    base_metrics.update(additional)
    
    return base_metrics