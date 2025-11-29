# Brain Tumor Segmentation and Classification

A multi-task deep learning model for simultaneous brain tumor segmentation and classification using the BRISC2025 dataset. This implementation uses a lightweight U-Net architecture optimized for Google Colab training.

## Overview

This project implements a dual-output neural network that performs:
- **Segmentation**: Pixel-level tumor boundary detection
- **Classification**: Tumor type identification across 4 categories

The model achieves strong performance with optimized training parameters designed for efficient Colab execution.

## Dataset

**BRISC 2025** (Brain Tumor MRI Dataset for Segmentation and Classification)

A high-quality, expert-annotated MRI dataset addressing common limitations in existing datasets like BraTS and Figshare. The dataset includes class imbalance corrections, comprehensive tumor coverage, and consistent radiologist-verified annotations.

**Dataset Specifications:**
- 6,000 T1-weighted MRI images with corresponding segmentation masks
- Four tumor classes: Glioma, Meningioma, Pituitary, and No Tumor
- Multiple anatomical views: axial, coronal, sagittal
- Expert-curated annotations by radiologists and physicians
- ArXiv preprint: https://arxiv.org/abs/2506.14318

**Tumor Classes:**
- **gl**: Glioma
- **me**: Meningioma
- **pi**: Pituitary
- **nt**: No Tumor

## Architecture

**Lightweight Multi-Task U-Net**

The model features a shared encoder with two specialized decoder branches:

**Encoder:**
- Three encoder blocks with progressively increasing filters (32, 64, 128)
- Max pooling for downsampling
- Batch normalization and ReLU activation

**Segmentation Branch:**
- Three decoder blocks with skip connections
- Transposed convolutions for upsampling
- Binary mask output with sigmoid activation

**Classification Branch:**
- Global average pooling on bottleneck features
- Two dense layers (128, 64 units) with dropout
- Softmax output for 4-class prediction

**Model Optimizations:**
- Reduced filter counts for faster training
- Smaller input resolution (128x128)
- Lightweight classification head
- Efficient skip connections

## Training Configuration

**Hyperparameters:**
```
Image size: 128x128
Batch size: 16
Epochs: 50
Learning rate: 5e-4
Validation split: 15%
```

**Loss Functions:**
- Segmentation: Dice loss
- Classification: Categorical cross-entropy

**Data Augmentation:**
- Random horizontal flipping
- Random brightness adjustment

## Results

The model demonstrates excellent performance across both tasks:

**Segmentation Metrics:**

| Metric | Score |
|--------|-------|
| Dice Coefficient | 0.7814 |
| IoU | 0.6975 |
| Pixel Accuracy | 0.9944 |
| Sensitivity | 0.7908 |
| Specificity | 0.9978 |

These metrics indicate strong tumor boundary detection with high precision and minimal false positives.

## Setup and Installation

**Requirements:**
```bash
tensorflow
numpy
opencv-python
scikit-learn
```

**Google Colab Setup:**

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Ensure dataset is organized in the following structure:
```
Brain Tumor Segmentation and Classification/
└── training_SupervisedLearning/
    └── brisc2025/
        ├── segmentation_task/
        │   └── train/
        │       ├── images/
        │       └── masks/
        └── classification_task/
```

3. Run the training script in a Colab notebook cell

## Usage

**Training:**

Simply execute the main script:
```python
python train.py
```

The training process will:
- Load and split the dataset
- Display class distribution statistics
- Build the multi-task model
- Train with automatic checkpointing
- Save the best model and training history

**Output Files:**

All outputs are saved to the checkpoints directory:
- Best model checkpoint based on validation loss
- Final trained model
- Training history in JSON format

**Training Monitoring:**

The custom callback displays epoch-by-epoch metrics:
- Segmentation loss and Dice coefficient
- Classification loss and accuracy
- Validation metrics for both tasks

## Model Performance

Training achieves convergence within 50 epochs with early stopping monitoring. The multi-task learning approach enables the model to leverage shared representations, improving both segmentation and classification performance.

**Key Performance Indicators:**
- High Dice coefficient indicates accurate tumor boundary detection
- Near-perfect specificity minimizes false tumor predictions
- Balanced sensitivity ensures most tumors are correctly identified
- Strong pixel accuracy reflects overall segmentation quality

## File Naming Convention

Dataset files follow the BRISC2025 format:
```
brisc2025_train_00001_gl_ax_t1.jpg
```

**Components:**
- Dataset identifier: `brisc2025`
- Split: `train/test`
- Index: `00001`
- Tumor code: `gl/me/pi/nt`
- View: `ax/co/sa` (axial/coronal/sagittal)
- Sequence: `t1` (T1-weighted)

## Optimization Features

This implementation includes several optimizations for Colab training:

- 4x faster processing with reduced image resolution
- 2x larger batch size for better GPU utilization
- 50% reduction in model parameters
- Simplified architecture with 3 encoder layers
- Efficient data generators with on-the-fly loading
- Stratified train-validation split

## Next Steps

After training completion:
1. Run validation analysis on the saved model
2. Evaluate performance on the test set
3. Visualize predictions on sample images
4. Export model for deployment or inference

## License

Please refer to the BRISC2025 dataset documentation for usage terms and conditions.

## Citation

If you use this code or the BRISC2025 dataset, please cite:
```bibtex
@article{brisc2025,
  title={BRISC 2025: Brain Tumor MRI Dataset for Segmentation and Classification},
  journal={ArXiv preprint},
  url={https://arxiv.org/abs/2506.14318},
  year={2025}
}
```

## Acknowledgments

Dataset curated and annotated by expert radiologists and physicians. Model architecture based on the U-Net framework adapted for multi-task learning.
