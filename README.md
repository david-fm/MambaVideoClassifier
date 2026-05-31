# Efficient CNN-Mamba Architectures for Real-Time WLASL Recognition on Edge Devices

This project implements a **MobileNetV3-Large + Selective State Space Model (Mamba)** architecture for real-time word-level American Sign Language (ASL) recognition on the WLASL dataset. The work explores the accuracy–efficiency trade-off required for edge deployment, moving from an initial unidirectional configuration (V1) to a refined pipeline (V2) with stronger regularisation and augmentation.

## Architecture

- **Spatial Encoder:** MobileNetV3-Large (ImageNet pretrained) → per-frame feature extraction (960-dim embeddings)
- **Temporal Decoder:** 2-layer selective SSM (Mamba, hidden dim 256, state dim 16) → linear-time sequence modelling
- **Classification Head:** Temporal average pooling + fully-connected layer → 300-class output

**Key Properties:**
- Linear temporal complexity: O(n)
- Fixed memory state during inference (no KV-cache growth)
- ~4.2M parameters, ~16 MB (FP32)
- 16 frames × 224 × 224 RGB input

## Dataset

**WLASL300** (WLASL subset with 300 most frequent classes):
- 5,117 video clips (Train: 3,549 | Val: 900 | Test: 668)
- 14–40 instances per class (mean: 17.1)
- Native 25 fps, pre-extracted to 25-frame uint8 `.npy` arrays for stable loading

## Project Structure

```
CV2_Project_2/
├── notebooks/
│   ├── mamba-video-classifier-v2.ipynb   # Final training & evaluation pipeline (V2)
│   └── mamba-video-classifier-v1.ipynb   # Initial configuration (V1)
├── data/ # downloaded from V2 Kaggle output and from WLASL dataset
│   ├── WLASL_v0.3.json                   # Dataset annotations
│   ├── wlasl300_class_distribution.csv # Class statistics
│   ├── test_results.json               # Final V2 test metrics
│   ├── frames_manifest.json              # Pre-extracted frame index
│   └── *.png                             # Visualisations (confusion matrix, training curves, failure analysis)
├── pyproject.toml                       # Project dependencies (uv/pip)
├── uv.lock                              # Locked dependency tree
├── .python-version                      # Python 3.12+
└── README.md                            # This file
```

## Quick Start

### 1. Setup (Local)

The project uses `uv` for Python environment management. If you have `uv` installed:

```bash
uv sync
source .venv/bin/activate
```

Required packages include: `torch`, `torchvision`, `opencv-python`, `scikit-learn`, `seaborn`, `matplotlib`, `pandas`, `numpy`, `notebook`.

### 2. Run the Notebook

Open `notebooks/mamba-video-classifier-v2.ipynb` in Jupyter or Kaggle and run all cells sequentially. The notebook contains the complete pipeline: data preprocessing, model definition, training, evaluation, and inference profiling.

### 3. Dataset Paths

Update the notebook configuration cell to point to your WLASL data:

```python
Config.JSON_PATH = 'data/WLASL_v0.3.json'
Config.VIDEO_ROOT = '<path_to_wlasl_videos>'
Config.FRAMES_ROOT = '<path_to_preextracted_npy_frames>'
```

The data loader transparently falls back to a secondary Kaggle source (`wlasl2000-resized`) if a clip is missing from the primary directory.

## Model Details

### Spatial Encoder
- **Backbone:** MobileNetV3-Large
- **Pretrained:** ImageNet-1K
- **Output:** 960-dim per-frame embeddings
- **Parameters:** ~5.4M (backbone only)
- **Frozen layers:** First 13 layers frozen during training to preserve low-level features

### Temporal Decoder
- **Type:** Selective State Space Model (Mamba)
- **Layers:** 2
- **Hidden dim:** 256
- **State dim:** 16
- **Parameters:** ~0.5M

### Total Model (V2)
- **Parameters:** 4.17M
- **Model size (FP32):** 15.92 MB
- **Input:** (B, 3, 16, 224, 224)
- **Output:** (B, 300)

## Training Configuration (V2)

- **Optimiser:** AdamW (backbone lr=1e-4, decoder/head lr=1e-3, weight_decay=1e-4)
- **LR Schedule:** ReduceLROnPlateau (patience=5, factor=0.3)
- **Loss:** Cross-Entropy with label smoothing (ε = 0.1)
- **Regularisation:** Dropout 0.6, gradient clipping (max_norm=1.0)
- **Batch size:** 16
- **Epochs:** Up to 100 (early stopping patience=10 on validation loss)
- **Data augmentation (training):**
  - Temporal jitter: random 16-frame window from 25 preloaded frames
  - Colour jitter: ±20% brightness/contrast/saturation, ±5% hue
  - Random horizontal flip (p=0.5)
  - Random crop (scale 0.85–1.0, then resize to 224×224)
  - Gaussian blur (p=0.3)
- **Frame sampling:** 16 frames uniformly subsampled from 25 pre-extracted frames
- **Normalisation:** ImageNet mean/std

## Results (WLASL300 Test Set)

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | **37.75%** |
| **Top-5 Accuracy** | **67.96%** |
| **Top-10 Accuracy** | **80.41%** |
| F1 (macro) | 0.347 |
| Precision (macro) | 0.362 |
| Recall (macro) | 0.377 |
| Parameters | 4.17M |
| Model size (FP32) | 15.92 MB |
| CPU inference latency | 240.89 ± 6.14 ms/video |
| CPU throughput | 4.2 videos/sec |

Training converged at epoch 48 (best validation Top-1 = 45.61%), with early stopping triggering at epoch 58.

## Evolution: V1 → V2

The initial executable configuration (V1) lacked temporal and colour augmentation, used a uniform learning rate, and fine-tuned the full MobileNetV3 backbone. V2 introduces stronger regularisation and augmentation, yielding a **+9.92 pp improvement in Top-1 accuracy**.

| Aspect | V1 (Initial) | V2 (Final) |
|--------|--------------|------------|
| Temporal jitter | None | Random 16-frame window |
| Colour jitter | None | Brightness/contrast/saturation/hue |
| Gaussian blur | None | p = 0.3 |
| Random crop | None | Scale 0.85–1.0 |
| Normalisation | [-1, 1] | ImageNet mean/std |
| MobileNetV3 freezing | None | First 13 layers frozen |
| Dropout | 0.5 | 0.6 |
| Label smoothing | None | 0.1 |
| Learning rate | Uniform 1e-3 | Differential (1e-4 / 1e-3) |
| Gradient clipping | None | max_norm = 1.0 |
| **Test Top-1** | **27.83%** | **37.75%** |
| **Test Top-5** | **58.98%** | **67.96%** |
| **Test Top-10** | **72.27%** | **80.41%** |

## Citation

If you use this code, please cite the core works below. The complete bibliography for the related work and thesis direction is available in the LaTeX report (`references.bib`).

```bibtex
@inproceedings{li2020wlasl,
  title={{WLASL}: A Large-Scale Dataset for Word-Level American Sign Language},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1856--1865},
  year={2020}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{howard2019searching,
  title={Searching for {MobileNetV3}},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={1314--1324},
  year={2019}
}

@inproceedings{li2024videomamba,
  title={VideoMamba: State Space Model for Efficient Video Understanding},
  author={Li, Kunchang and Li, Xinhao and Wang, Yi and He, Yinan and Wang, Yali and Wang, Limin and Qiao, Yu},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## License

This project is for academic and educational purposes. The WLASL dataset is licensed under C-UDA.

## Acknowledgments

- WLASL dataset by Li et al. (2020)
- Mamba by Gu & Dao (2023)
- MobileNetV3 by Howard et al. (2019)
