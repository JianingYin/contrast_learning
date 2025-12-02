# Contrastive Learning-Based Alzheimer's Disease Classification Model

[English Version](https://github.com/JianingYin/contrast_learning/blob/master/README.en.md) | [中文版](https://github.com/JianingYin/contrast_learning/blob/master/README.md)

## Project Overview

This project implements a deep learning and contrastive learning-based framework for Alzheimer's Disease (AD) classification. By fusing whole-brain and hippocampal MRI data with wavelet transform and supervised contrastive learning methods, the model achieves high-accuracy multi-class diagnosis (AD, MCI, CN).

## Methodology

### Core Innovations
- **Dual-Branch Fusion Architecture**: Simultaneously processes whole-brain and hippocampal MRI data to capture both local and global features
- **Wavelet Transform Preprocessing**: Utilizes wavelet transform to extract low-frequency components, enhancing critical medical imaging features
- **Supervised Contrastive Learning**: Optimizes feature space distribution through SupConLoss to improve classification performance
- **Attention Mechanisms**: Uses SEBlock and ECA_Module to adaptively weight important features

### Model Architecture

```
FullModel
├── DeepFusionNetV2 (Backbone Network)
│   ├── WaveletLowpass (Wavelet Transform Module)
│   ├── Whole-Brain Feature Extraction Branch
│   ├── Hippocampus Feature Extraction Branch
│   └── GatedFusion (Gated Fusion Module)
├── VGGStyleEncoder (Feature Encoder)
├── ProjectionHead (Contrastive Learning Projection Head)
└── ClassificationHead (Classification Head)
```

## Dataset

### Data Structure
- **Whole-brain MRI data**: Located in the `datasets/NIFTI5` directory
- **Hippocampal MRI data**: Located in the `datasets/hippdata` directory
- **Data format**: NIfTI (.nii/.nii.gz)

### Classification Task
- **Classes**: 3 categories (Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), Cognitively Normal (CN))
- **Evaluation Method**: 5-fold cross-validation with OASIS independent validation set

## Installation

### Environment Requirements
- Python 3.8+
- PyTorch 1.8+
- nibabel
- numpy
- scikit-learn
- pywavelets
- matplotlib
- tqdm

### Install Dependencies
```bash
pip install torch torchvision nibabel numpy scikit-learn pywavelets matplotlib tqdm
```

## Usage

### Configuration
Modify parameters in the `config.py` file:
```python
# Hyperparameter settings
device = 'cuda'  # or 'cpu'
batch_size = 8
lr = 1e-4
epochs = 50
supcon_weight = 0.1

# Data paths
brain_root = 'path/to/NIFTI5/all'
hippocampus_root = 'path/to/hippdata/all'
```

### Training the Model
Run the main script for training and evaluation:
```bash
python main2.py --n_folds 5 --batch_size 8 --device cuda
```

### Optional Parameters
- `--n_folds`: Number of cross-validation folds (default: 5)
- `--batch_size`: Batch size
- `--device`: Execution device (cuda or cpu)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate

## Experimental Results

### Key Performance Metrics
- **Best Validation Accuracy**: 0.9904 (5-fold average)
- **AUC Value**: >0.98
- **F1 Score**: >0.97

### Experimental Output
All experimental results are saved in the `runs/experiment_YYYYMMDD_HHMMSS/` directory, including:
- Training logs and model weights for each fold
- Visualization results such as confusion matrices, ROC curves, and PR curves
- 5-fold cross-validation metrics statistics and confidence intervals
- Results on the OASIS independent validation set

## File Structure

```
CONTRAST_LEARNING-master/
├── main2.py                # Main training and evaluation script
├── config.py               # Configuration parameters
├── datasets/               # Dataset processing modules
│   ├── datasets_class.py   # Dataset class definitions
│   └── datasets_5fold.py   # 5-fold data splitting
├── models/                 # Model definitions
│   └── model/
│       └── modelV24.py     # Final model implementation
├── utils/                  # Utility functions
│   ├── supcon_loss.py      # Contrastive learning loss function
│   ├── evaluation_metrics.py  # Evaluation metrics
│   └── painter.py          # Result visualization
└── runs/                   # Experimental results storage directory
```

## Citation

If you use the code or methods from this project, please cite our paper:

```
@article{your_paper_reference,
  title={Dual-Branch Contrastive Learning Method for Alzheimer's Disease Classification},
  author={Author Names},
  journal={Journal Name},
  year={2024},
  volume={Volume Number},
  number={Issue Number},
  pages={Page Numbers},
  publisher={Publisher}
}
```

## Contact Information

For questions or suggestions, please contact the project maintainer: [Your Contact Information]