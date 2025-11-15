# HAM10000 Skin Lesion Classification
**High-Accuracy Dermatology AI with Explainable Predictions**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Validation_Accuracy-94.21%25-brightgreen.svg)]()

## 🎯 Project Overview
Advanced deep learning solution for HAM10000 skin lesion classification achieving **94.21% validation accuracy** with a single model. Target: **97%+ with ensemble**.

**Team:** Aditya Raj, Aryan Roy, Adarsh Kumar Pradhan

## 📊 Current Results
| Model | Image Size | Params | Val Accuracy | Training Time |
|-------|-----------|--------|--------------|---------------|
| **EfficientNetV2-RW-S** | 256×256 | 24M | **94.21%** | ~1.5 hours |
| ConvNeXt-Tiny | 256×256 | 28M | Training | ~1.5 hours |
| **Ensemble (Expected)** | 256×256 | - | **~96-97%** | - |

## 📁 Project Structure
```
dataverse_files/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── train_ultra_fast.py           # Fast training script (RECOMMENDED)
├── evaluate_ensemble.py          # Ensemble evaluation
│
├── src/                          # Core source code
│   ├── data/
│   │   └── dataset.py           # HAM10000Dataset class
│   ├── models/
│   │   └── model.py             # Model architectures
│   ├── training/
│   │   ├── losses.py            # Focal Loss
│   │   └── metrics.py           # Evaluation metrics
│   ├── explainability/
│   │   └── gradcam.py           # Grad-CAM visualization
│   ├── data_preparation.py      # Dataset analyzer
│   └── train_optimized.py       # Full-featured trainer
│
├── data/                         # Dataset files
│   ├── raw/                     # Original data & metadata
│   └── processed/               # train.csv, val.csv (DO NOT DELETE)
│
├── checkpoints/                  # Trained models
│   ├── best_model_fast.pth      # EfficientNetV2-RW-S (94.21%)
│   └── best_model_convnext.pth  # ConvNeXt-Tiny
│
├── scripts/                      # Utility scripts
│   ├── check_model.py           # Inspect checkpoints
│   ├── diagnose_speed.py        # Performance profiling
│   └── profile_training.py      # Training profiler
│
├── docs/                         # Documentation
│   ├── QUICKSTART.md
│   ├── SETUP_GUIDE.md
│   ├── SPEED_OPTIMIZATION.md
│   └── ...
│
└── outputs/                      # Training logs & visualizations
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data (Already Done ✅)
```bash
python src/data_preparation.py
# Creates processed/train.csv and processed/val.csv
```

### 3. Train Model
```bash
# Fast training (RECOMMENDED - 1.5 hours)
python train_ultra_fast.py --model efficientnetv2_rw_s --img_size 256 --batch_size 32 --epochs 30

# Or full training with more options
python src/train_optimized.py --model efficientnetv2_rw_m --img_size 384 --batch_size 16 --epochs 50
```

### 4. Evaluate Ensemble
```bash
python evaluate_ensemble.py
# Combines multiple models for higher accuracy
```

## 🎓 Dataset Information

### HAM10000 - 7 Skin Lesion Classes
1. **mel** - Melanoma (1,113 images)
2. **nv** - Melanocytic nevi (6,705 images) ⚠️ Highly imbalanced
3. **bkl** - Benign keratosis (1,099 images)
4. **bcc** - Basal cell carcinoma (514 images)
5. **akiec** - Actinic keratoses (327 images)
6. **vasc** - Vascular lesions (142 images)
7. **df** - Dermatofibroma (115 images)

**Total:** 10,015 images | **Split:** 8,512 train / 1,503 validation (stratified 85/15)

## 🔬 Key Technical Features

### Data Processing
- ✅ Stratified train/val split maintaining class distribution
- ✅ Lightweight augmentations (HFlip, VFlip, Rotate90, Affine, ColorJitter)
- ✅ ImageNet normalization for transfer learning

### Model Architecture
- ✅ Pre-trained backbones from `timm` (ImageNet-1k weights)
- ✅ Custom classifier head with dropout (0.2)
- ✅ Support for EfficientNetV2, ConvNeXt, Swin, ViT

### Training Optimizations
- ✅ **Focal Loss** (γ=2.0) for class imbalance
- ✅ **Mixed Precision (FP16)** for 2x speedup
- ✅ **AdamW optimizer** with weight decay (0.01)
- ✅ **CosineAnnealing LR** scheduler
- ✅ Automatic checkpointing (saves best model)

### Performance Optimizations
- ✅ Reduced image size (256×256) for speed
- ✅ Larger batch size (32) for GPU efficiency
- ✅ Non-blocking data transfer
- ✅ Efficient DataLoader settings

## 💻 Hardware Requirements

**Minimum (Tested):**
- GPU: NVIDIA RTX 3060 Laptop (6GB VRAM)
- RAM: 16GB
- Storage: 20GB

**Recommended:**
- GPU: NVIDIA RTX 3070+ (8GB+ VRAM)
- RAM: 32GB
- Storage: 50GB

## 📈 Training Progress & Strategy

### Achieved: Single Model
- **EfficientNetV2-RW-S:** 94.21% validation accuracy
- Training time: ~1.5 hours on RTX 3060
- Model size: 24M parameters

### In Progress: Ensemble
- **ConvNeXt-Tiny:** Training (~94-95% expected)
- Expected ensemble gain: +2-3%
- Target ensemble accuracy: **96-97%**

### To Reach 98%+:
1. ✅ Train 2 diverse models (EfficientNet + ConvNeXt)
2. ⏳ Evaluate weighted ensemble
3. 🔲 Add Test-Time Augmentation (TTA) → +1-2%
4. 🔲 Optional: Train 3rd model (Swin/ViT) if needed

## 🛠️ Utility Scripts

```bash
# Check saved model details
python scripts/check_model.py

# Profile training performance
python scripts/profile_training.py

# Diagnose speed bottlenecks
python scripts/diagnose_speed.py
```

## 📚 Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Fast setup guide
- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Detailed installation
- **[SPEED_OPTIMIZATION.md](docs/SPEED_OPTIMIZATION.md)** - Performance tuning
- **[TRAINING_CONFIGS.md](docs/TRAINING_CONFIGS.md)** - Config options

## 🔍 Explainability (Grad-CAM)

```python
from src.explainability.gradcam import GradCAM

# Generate heatmap showing what the model focuses on
gradcam = GradCAM(model, target_layer)
heatmap = gradcam.generate_cam(image, target_class)
```

## 📊 Evaluation Metrics

```python
from src.training.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred, y_prob)
# Returns: accuracy, f1_macro, precision_macro, recall_macro, confusion_matrix
```

## 🤝 Contributing

This is an academic project. For questions or collaboration:
- Aditya Raj
- Aryan Roy
- Adarsh Kumar Pradhan

## 📄 License

Academic use only - HAM10000 dataset terms apply

## 🙏 Acknowledgments

- HAM10000 Dataset by Tschandl et al.
- `timm` library for pre-trained models
- PyTorch team for deep learning framework

## 📞 Support

For issues or questions:
1. Check `docs/` folder for detailed guides
2. Review training logs in `outputs/`
3. Run diagnostic scripts in `scripts/`

---

**Last Updated:** November 4, 2025  
**Project Status:** ✅ Working | 🎯 Target: 98%+ accuracy
