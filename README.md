<div align="center">
# 基于对比学习的阿尔茨海默病分类模型

[中文版](https://github.com/JianingYin/contrast_learning/blob/master/README.md) 
[English Version](https://github.com/JianingYin/contrast_learning/blob/master/README.en.md)

</div>

## 项目概述

本项目实现了一个基于深度学习和对比学习的阿尔茨海默病(AD)分类框架，通过融合全脑和海马体MRI数据，结合小波变换和监督对比学习方法，实现高精度的多分类诊断（AD、MCI、CN）。

## 方法简介

### 核心创新点
- **双分支融合架构**：同时处理全脑和海马体MRI数据，捕捉局部和全局特征
- **小波变换预处理**：利用小波变换提取低频分量，增强关键医学影像特征
- **监督对比学习**：通过SupConLoss优化特征空间分布，提高分类性能
- **注意力机制**：使用SEBlock和ECA_Module自适应加权重要特征

### 模型架构

```
FullModel
├── DeepFusionNetV2 (主干网络)
│   ├── WaveletLowpass (小波变换模块)
│   ├── 全脑特征提取分支
│   ├── 海马体特征提取分支
│   └── GatedFusion (门控融合模块)
├── VGGStyleEncoder (特征编码器)
├── ProjectionHead (对比学习投影头)
└── ClassificationHead (分类头)
```

## 数据集

### 数据结构
- **全脑MRI数据**：位于`datasets/NIFTI5`目录
- **海马体MRI数据**：位于`datasets/hippdata`目录
- **数据格式**：NIfTI (.nii/.nii.gz)

### 分类任务
- **类别**：3类（阿尔茨海默病(AD)、轻度认知障碍(MCI)、认知正常(CN)）
- **评估方式**：5折交叉验证，OASIS独立验证集

## 安装说明

### 环境要求
- Python 3.8+
- PyTorch 1.8+
- nibabel
- numpy
- scikit-learn
- pywavelets
- matplotlib
- tqdm

### 安装依赖
```bash
pip install torch torchvision nibabel numpy scikit-learn pywavelets matplotlib tqdm
```

## 使用方法

### 配置
修改`config.py`文件中的参数：
```python
# 超参数设置
device = 'cuda'  # 或 'cpu'
batch_size = 8
lr = 1e-4
epochs = 50
supcon_weight = 0.1

# 数据路径
brain_root = 'path/to/NIFTI5/all'
hippocampus_root = 'path/to/hippdata/all'
```

### 训练模型
运行主脚本进行训练和评估：
```bash
python main2.py --n_folds 5 --batch_size 8 --device cuda
```

### 可选参数
- `--n_folds`：交叉验证折数（默认5）
- `--batch_size`：批次大小
- `--device`：运行设备（cuda或cpu）
- `--epochs`：训练轮数
- `--lr`：学习率

## 实验结果

### 主要性能指标
- **最佳验证准确率**：0.9904（5折平均）
- **AUC值**：>0.98
- **F1分数**：>0.97

### 实验输出
所有实验结果保存在`runs/experiment_YYYYMMDD_HHMMSS/`目录下，包括：
- 每折的训练日志和模型权重
- 混淆矩阵、ROC曲线、PR曲线等可视化结果
- 5折交叉验证指标统计和置信区间
- OASIS独立验证集结果

## 文件结构

```
CONTRAST_LEARNING-master/
├── main2.py                # 主训练和评估脚本
├── config.py               # 配置参数
├── datasets/               # 数据集处理模块
│   ├── datasets_class.py   # 数据集类定义
│   └── datasets_5fold.py   # 5折数据划分
├── models/                 # 模型定义
│   └── model/
│       └── modelV24.py     # 最终模型实现
├── utils/                  # 工具函数
│   ├── supcon_loss.py      # 对比学习损失函数
│   ├── evaluation_metrics.py  # 评估指标
│   └── painter.py          # 结果可视化
└── runs/                   # 实验结果保存目录
```

## 引用

如果您使用本项目的代码或方法，请引用我们的论文：

```
@article{your_paper_reference,
  title={阿尔茨海默病分类的双分支对比学习方法},
  author={作者姓名},
  journal={期刊名称},
  year={2024},
  volume={卷号},
  number={期号},
  pages={页码},
  publisher={出版社}
}
```


## 联系信息

如有问题或建议，请联系项目维护者：[您的联系方式]