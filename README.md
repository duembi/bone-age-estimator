# Pediatric Bone Age Estimation via Deep Learning

A multi-architecture deep learning comparison for automated skeletal maturity assessment, trained on the RSNA Pediatric Bone Age Challenge 2017 dataset.

## Results

| Model | MAE (months) | R² |
|-------|-------------|-----|
| **EfficientNetB0** | **7.66** | **0.9416** |
| DenseNet121 | 7.85 | 0.9368 |
| InceptionV3 | 7.99 | 0.9355 |
| Xception | 8.03 | 0.9333 |
| VGG16 | 8.21 | 0.9315 |
| ResNet50 | 8.30 | 0.9347 |
| EfficientNetB3 | 8.74 | 0.9260 |
| **Ensemble** | **~7.5** | — |

## Repository Structure

```
├── final-final.ipynb        # Main training & evaluation notebook (Kaggle)
├── report_ieee.tex          # IEEE-format LaTeX paper
├── figures/                 # All result plots and visualizations
└── results/
    ├── multi_model_train.py       # Training pipeline (7 CNN architectures)
    ├── comprehensive_metrics.py   # Full evaluation suite (21 metrics)
    ├── eda_and_visualization.py   # EDA plots
    ├── patient_predict.py         # Ensemble inference & clinical report
    ├── all_models_metrics.csv     # Full metrics table
    ├── model_comparison_summary.csv
    └── history_*.csv              # Per-model training histories
```

## Models

VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNet-B0, EfficientNet-B3, Xception — all trained under identical hyperparameters with ImageNet pretraining.

## Preprocessing

- CLAHE (clipLimit=2.0, 8×8 tile grid)
- Aspect-ratio preserving zero-padding
- Gender-conditioned regression head

## Dataset

[RSNA Pediatric Bone Age Challenge 2017](https://www.kaggle.com/datasets/kmader/rsna-bone-age) — 12,611 hand radiographs.

## Paper

IEEE-format report available in `report_ieee.tex`. Compile with pdfLaTeX or upload to [Overleaf](https://www.overleaf.com).

## Course

Pattern Recognition — Altinbas University, Spring 2025–26
