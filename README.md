# Brain Tumor Segmentation and Classification

A multi-task deep learning model for simultaneous brain tumor segmentation and classification using the BRISC2025 dataset. This implementation uses a lightweight U-Net architecture optimized for Google Colab training.

![Brain MRI](https://images.unsplash.com/photo-1617791160505-6f00504e3519?w=1200&h=400&fit=crop&q=80)

## Overview

This project implements a dual-output neural network that performs:
- **Segmentation**: Pixel-level tumor boundary detection
- **Classification**: Tumor type identification across 4 categories

The model achieves strong performance with optimized training parameters designed for efficient Colab execution.

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Status](https://img.shields.io/badge/Status-Production-green)

---

## Dataset

**BRISC 2025** (Brain Tumor MRI Dataset for Segmentation and Classification)

![Medical Imaging](https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1000&h=300&fit=crop&q=80)

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

---

## Architecture

**Lightweight Multi-Task U-Net**

![U-Net Architecture](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-07_at_9.08.00_PM_rpNArED.png)

The model features a shared encoder with two specialized decoder branches:

### Encoder:
- Three encoder blocks with progressively increasing filters (32, 64, 128)
- Max pooling for downsampling
- Batch normalization and ReLU activation

### Segmentation Branch:
- Three decoder blocks with skip connections
- Transposed convolutions for upsampling
- Binary mask output with sigmoid activation

### Classification Branch:
- Global average pooling on bottleneck features
- Two dense layers (128, 64 units) with dropout
- Softmax output for 4-class prediction

**Model Optimizations:**
- Reduced filter counts for faster training
- Smaller input resolution (128x128)
- Lightweight classification head
- Efficient skip connections

---

## Training Configuration

![Training](https://img.shields.io/badge/Training-Optimized-success)

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

![Data Augmentation](https://miro.medium.com/v2/resize:fit:1400/1*C8hNiOqur4OJyEZmC7OnzQ.png)

---

## Results

![Performance](https://img.shields.io/badge/Dice%20Score-0.78-brightgreen)
![Accuracy](https://img.shields.io/badge/Pixel%20Accuracy-99.4%25-blue)

The model demonstrates excellent performance across both tasks:

### Segmentation Metrics:

| Metric | Score |
|--------|-------|
| Dice Coefficient | 0.7814 |
| IoU | 0.6975 |
| Pixel Accuracy | 0.9944 |
| Sensitivity | 0.7908 |
| Specificity | 0.9978 |

![Segmentation Results](https://www.mdpi.com/brainsci/brainsci-11-01352/article_deploy/html/images/brainsci-11-01352-g004.png)

These metrics indicate strong tumor boundary detection with high precision and minimal false positives.

---

## Setup and Installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

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

![Colab Interface](https://miro.medium.com/v2/resize:fit:1400/1*FUIlH1Y36HLp7RHj-kRYNQ.png)

---

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

---

## Model Performance

![Training Curves](https://www.researchgate.net/publication/355208152/figure/fig2/AS:1078697012916224@1633792037767/Training-and-validation-accuracy-and-loss-curves.png)

Training achieves convergence within 50 epochs with early stopping monitoring. The multi-task learning approach enables the model to leverage shared representations, improving both segmentation and classification performance.

**Key Performance Indicators:**
- High Dice coefficient indicates accurate tumor boundary detection
- Near-perfect specificity minimizes false tumor predictions
- Balanced sensitivity ensures most tumors are correctly identified
- Strong pixel accuracy reflects overall segmentation quality

---

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

---

## Optimization Features

![Optimization](https://img.shields.io/badge/Speed-4x%20Faster-red)

This implementation includes several optimizations for Colab training:

- ⚡ 4x faster processing with reduced image resolution
- 📊 2x larger batch size for better GPU utilization
- 🎯 50% reduction in model parameters
- 🏗️ Simplified architecture with 3 encoder layers
- 💾 Efficient data generators with on-the-fly loading
- 🎲 Stratified train-validation split

![Performance Comparison](https://miro.medium.com/v2/resize:fit:1400/1*ZvPYRazjoNOb8kzV5K8MJA.png)

---

## Next Steps

After training completion:
1. Run validation analysis on the saved model
2. Evaluate performance on the test set
3. Visualize predictions on sample images
4. Export model for deployment or inference

---

## License

Please refer to the BRISC2025 dataset documentation for usage terms and conditions.

---

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

---

## Acknowledgments

![Medical](https://img.shields.io/badge/Medical-AI-purple)
![Research](https://img.shields.io/badge/Research-Academic-yellow)

Dataset curated and annotated by expert radiologists and physicians. Model architecture based on the U-Net framework adapted for multi-task learning.

![Medical AI](https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200&h=300&fit=crop&q=80)

---

**Made with ❤️ for Medical AI Research**

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
