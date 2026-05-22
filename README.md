# WLASL Sign Language Recognition with CNN-Mamba

This project implements a **MobileNetV3-Large + Selective State Space Model (Mamba)** architecture for real-time word-level American Sign Language (ASL) recognition on the WLASL dataset.

## Architecture

- **Spatial Encoder:** MobileNetV3-Large (ImageNet pretrained) → per-frame feature extraction
- **Temporal Decoder:** 2-layer selective SSM (Mamba) → linear-time sequence modeling
- **Classification Head:** Temporal pooling + fully-connected layer

**Key Properties:**
- Linear temporal complexity: O(n)
- Fixed memory state during inference (no KV cache growth)
- ~6M parameters, suitable for edge deployment
- 32 frames × 224 × 224 RGB input

## Dataset

**WLASL300** (WLASL subset with 300 most frequent classes):
- 5,117 video clips
- Train: 3,549 | Val: 900 | Test: 668
- 14–40 instances per class (mean: 17.1)

## Project Structure

```
CV2_Project/
├── src/
│   ├── data_analysis.py      # Dataset exploration and visualization
│   ├── data_loader.py        # PyTorch dataset and data loaders
│   ├── model.py              # CNN-Mamba model architecture
│   ├── train.py              # Training and evaluation utilities
│   └── utils.py              # Helper functions (plots, metrics, etc.)
├── notebooks/
│   └── WLASL_CNN_Mamba.ipynb # Main Kaggle notebook (complete pipeline)
├── data/                     # Dataset directory (not included)
└── README.md
```

## Quick Start

### 1. Setup (Kaggle)

Upload the WLASL dataset to Kaggle and update the paths in the notebook:
```python
Config.JSON_PATH = '/kaggle/input/wlasl/WLASL_v0.3.json'
Config.VIDEO_ROOT = '/kaggle/input/wlasl/videos'
```

### 2. Run the Notebook

Open `notebooks/WLASL_CNN_Mamba.ipynb` in Kaggle and run all cells sequentially.

### 3. Local Development

```bash
# Install dependencies
pip install torch torchvision opencv-python scikit-learn seaborn matplotlib pandas numpy

# Run data analysis
python src/data_analysis.py

# Test model
python src/model.py

# Test data loader
python src/data_loader.py
```

## Model Details

### Spatial Encoder
- **Backbone:** MobileNetV3-Large
- **Pretrained:** ImageNet-1K
- **Output:** 960-dim per-frame embeddings
- **Parameters:** ~5.4M

### Temporal Decoder
- **Type:** Selective State Space Model (Mamba)
- **Layers:** 2
- **Hidden dim:** 256
- **State dim:** 16
- **Parameters:** ~0.5M

### Total Model
- **Parameters:** ~6M
- **Model size (FP32):** ~24 MB
- **Input:** (B, 3, 32, 224, 224)
- **Output:** (B, 300)

## Training Configuration

- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **LR Schedule:** ReduceLROnPlateau (patience=5, factor=0.3)
- **Loss:** Cross-Entropy
- **Batch size:** 32
- **Epochs:** Up to 100 (early stopping patience=10)
- **Data augmentation:** Random horizontal flip, random crop
- **Frame sampling:** 32 frames uniformly sampled

## Expected Results (WLASL300)

Based on architecture design and comparable methods:
- **Top-1 Accuracy:** ~25-35%
- **Top-5 Accuracy:** ~45-55%
- **Top-10 Accuracy:** ~55-65%
- **Inference time:** ~20-50 ms/video (RTX 3090, batch=1)

## Comparison with Baselines

| Model | Params | Top-1 | Top-5 | FLOPs |
|-------|--------|-------|-------|-------|
| I3D (Li et al. 2020) | ~12M | ~32% | ~57% | ~150G |
| **Ours (CNN-Mamba)** | **~6M** | **TBD** | **TBD** | **~10G** |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{li2020wlasl,
  title={WLASL: A Large-Scale Dataset for Word-Level American Sign Language},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={CVPR},
  year={2020}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## License

This project is for academic and educational purposes. The WLASL dataset is licensed under C-UDA.

## Acknowledgments

- WLASL dataset by Li et al. (2020)
- Mamba by Gu & Dao (2023)
- MobileNetV3 by Howard et al. (2019)
