# CNN-Based SAR Vehicle Target Recognition

A modular, documented PyTorch implementation for 3-class Synthetic Aperture Radar (SAR) vehicle target recognition (`BMP2`, `BTR70`, `T72`) on the MSTAR dataset, refactored from our undergraduate *Introduction to Deep Learning* course project at Northwestern Polytechnical University.

## Highlights
- **Network Architecture (`ConvN`)**: Two 2D convolutional blocks (`Conv2d + BatchNorm2d + ReLU`), spatial max-pooling, and four fully connected layers (`512 -> 128 -> 64 -> 32 -> 3`) with `BatchNorm1d` and `log_softmax` output.
- **Reproducible Split**: Fixed-seed 70% / 30% train-validation split (`SARTrainDataset`) with random crop augmentation (`32x32`).
- **Historical Coursework Benchmark**: The original 2022 coursework report recorded approximately **96% validation accuracy** on the 70/30 split (with held-out test labels evaluated by the course instructors); this refactored repository preserves the modular architecture and includes unit tests (`tests/test_model.py`).

## Repository Structure

```
sar-vehicle-cnn/
├── src/
│   ├── __init__.py
│   ├── dataset.py      # SARTrainDataset (70/30 split) & TestDataset (unlabeled inference)
│   ├── model.py        # ConvN 2D CNN architecture with layer-by-layer shape annotations
│   ├── train.py        # Training and validation loop (Adam + StepLR)
│   └── infer.py        # Test-set inference and label export (result.txt)
├── tests/
│   └── test_model.py   # Unit tests for model forward pass and dataset loader
├── data/
│   └── README.md       # Expected MSTAR list/image format
├── requirements.txt
└── LICENSE
```

## Team Contributions (Two-Student Project: Chi Zhang & Meng Li)
- **Chi Zhang**:
  - Implemented `TestDataset` for unlabeled SAR test-set loading (`src/dataset.py`).
  - Designed and implemented the `ConvN` CNN architecture and forward pass (`src/model.py`).
  - Implemented the backpropagation training loop (`src/train.py`) and test-set inference/label-export pipeline (`src/infer.py`).
  - Co-authored the final experimental report.
- **Meng Li**:
  - Implemented the training/validation dataset split (`SARTrainDataset`), validation loop, and transform/optimizer configuration.

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests (no dataset required)
python -m unittest discover -s tests -v

# 3. Train and evaluate (requires MSTAR data in data/)
python -m src.train --train-txt data/train.txt --epochs 100 --batch-size 100

# 4. Run inference on unlabeled test set
python -m src.infer --test-txt data/test.txt --weights checkpoints/best_model.pth --out result.txt
```

