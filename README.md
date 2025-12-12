# Lab 2: Comparative Analysis of Vision Architectures

## 📋 Project Overview
This laboratory explores and compares multiple computer vision architectures on the MNIST dataset:
- Custom CNN implementation
- Faster R-CNN adaptation
- Vision Transformer (ViT) from scratch
- Fine-tuning of pre-trained models (VGG16, AlexNet)

**Course**: Deep Learning - Master MBD  
**Institution**: Université Abdelmalek Essaadi, Faculté des Sciences et techniques de Tanger  
**Professor**: Pr. ELACHAK LOTFI

## 🎯 Objectives
1. Implement and compare CNN and Faster R-CNN architectures
2. Build a Vision Transformer from scratch following the original paper
3. Fine-tune pre-trained models for transfer learning
4. Perform comprehensive comparative analysis using multiple metrics
5. Understand architectural trade-offs for different vision tasks

## 📊 Results Summary

### Performance Comparison
| Model | Test Accuracy | F1-Score | Training Time | Parameters |
|-------|---------------|----------|---------------|------------|
| **CNN (Custom)** | **99.03%** | 0.990 | 114s | 0.5M |
| **VGG16 (Fine-tuned)** | 99.43% | 0.994 | 3384s | 134.3M |
| **AlexNet (Fine-tuned)** | 99.33% | 0.993 | 838s | 57.0M |
| **ViT (From Scratch)** | 75.87% | ~0.76 | ~1689s | ~10k |

### Key Findings
- **CNNs dominate on MNIST**: Achieving >99% accuracy with minimal training time
- **ViTs struggle with small datasets**: Only 76% accuracy despite architectural novelty
- **Pre-trained models are overkill**: Excellent accuracy but extreme computational cost
- **Architecture-choice depends on data**: No universal best solution

## 🏗️ Implementation Details

### 1. CNN Architecture
- Two convolutional layers (32 and 64 filters)
- Max pooling and dropout regularization
- Optimized for MNIST's 28×28 grayscale images

### 2. Vision Transformer (ViT)
- Implemented from scratch following Dosovitskiy et al. 2020
- 7×7 patches from 28×28 images (49 patches + [CLS] token)
- 2 transformer blocks with 2 attention heads
- Sinusoidal positional encodings

### 3. Fine-tuning Pipeline
- Feature extraction freezing for pre-trained models
- Classifier replacement for 10-class MNIST
- Differential learning rates (lower for features, higher for classifier)



### Prerequisites
```bash
pip install torch torchvision numpy pandas matplotlib tqdm scikit-learn
