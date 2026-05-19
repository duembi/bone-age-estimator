# Pediatric Bone Age Estimation System

An automated deep learning system that estimates pediatric bone age from a single hand-wrist radiograph. Submit an X-ray image and the patient's gender — the system returns a bone age estimate in months along with a structured clinical report.

> **Course:** Pattern Recognition — Ankara University, Computer Engineering, Spring 2025–26  
> **Dataset:** [RSNA Pediatric Bone Age Challenge 2017](https://www.kaggle.com/datasets/kmader/rsna-bone-age) — 12,611 hand radiographs

---

## How It Works

```
Input:  Hand-wrist X-ray image  +  Patient gender
           ↓
    CLAHE preprocessing
           ↓
    EfficientNet-B0 backbone (gender-conditioned)
           ↓
    7-model ensemble inference
           ↓
Output: Bone age (months)  +  Clinical PDF report
```

The system requires **no radiologist intervention**. From image submission to report delivery takes a few seconds.

---

## Quick Start

### 1. Install dependencies

```bash
pip install tensorflow opencv-python numpy pandas matplotlib reportlab scikit-learn
```

### 2. Predict bone age for a single patient

```python
from patient_predict import predict_bone_age

result = predict_bone_age(
    image_path="patient_xray.png",
    gender=1,          # 1 = male, 0 = female
    models_dir="models/"
)

print(f"Bone Age:   {result['bone_age_months']:.1f} months")
print(f"95% CI:     ± {result['confidence_interval']:.1f} months")
print(f"Growth Stage: {result['tanner_stage']}")
```

### 3. Generate clinical PDF report

```python
from generate_report import generate_clinical_report

generate_clinical_report(
    image_path="patient_xray.png",
    gender=1,
    output_pdf="report.pdf"
)
```

The report includes:
- Bone age estimate with 95% confidence interval
- Tanner growth stage classification
- Skeletal maturity progress (0–228 months scale)
- Per-model prediction breakdown (ensemble transparency)
- Estimated remaining bone development time

---

## System Performance

| Model | MAE (months) ↓ | RMSE ↓ | R² ↑ |
|-------|---------------|--------|------|
| **EfficientNetB0** *(system backbone)* | **7.66** | **9.97** | **0.9416** |
| DenseNet121 | 7.85 | 10.37 | 0.9368 |
| InceptionV3 | 7.99 | 10.49 | 0.9355 |
| Xception | 8.03 | 10.66 | 0.9333 |
| ResNet50 | 8.30 | 10.55 | 0.9347 |
| VGG16 | 8.21 | 10.80 | 0.9315 |
| EfficientNetB3 | 8.74 | 11.23 | 0.9260 |
| **Ensemble (all 7)** | **~7.5** | — | — |

EfficientNet-B0 was selected as the primary backbone. All seven models participate in ensemble inference, reducing prediction variance to ±1.6 months.

Manual clinical assessment (Greulich–Pyle atlas) has an inter-rater variability of 9–12 months. The system achieves a **~30% improvement** over this baseline.

---

## Repository Structure

```
├── final-final.ipynb            # Full training & evaluation pipeline (Kaggle)
├── report_ieee.tex              # IEEE-format paper (LaTeX)
├── figures/                     # Result plots and visualizations
│   ├── 01_eda_metadata.png
│   ├── 02_eda_images_clahe.png
│   ├── 07_model_comparison_bars.png
│   ├── 08_radar_chart.png
│   ├── 09_all_models_scatter.png
│   ├── 10_metrics_heatmap.png
│   ├── 11_age_group_mae.png
│   ├── 12_training_histories.png
│   ├── 13_gender_mae.png
│   └── patient_report.png
└── results/
    ├── multi_model_train.py         # Training pipeline (7 CNN architectures)
    ├── comprehensive_metrics.py     # Evaluation suite (21 metrics)
    ├── eda_and_visualization.py     # EDA & visualization scripts
    ├── patient_predict.py           # Ensemble inference & clinical report
    ├── generate_report.py           # PDF report generator
    ├── all_models_metrics.csv       # Full metrics table
    ├── model_comparison_summary.csv
    └── history_*.csv                # Per-model training histories
```

---

## Architecture

### Preprocessing Pipeline
1. **CLAHE** — Contrast-Limited Adaptive Histogram Equalization (`clipLimit=2.0`, `8×8` tile grid) to enhance epiphyseal plate visibility
2. **Square padding** — Zero-pad to square canvas, preserving aspect ratio
3. **Resize & channel replication** — Resize to model input size, replicate to 3 channels for ImageNet compatibility

### Model Architecture
```
ImageNet-pretrained backbone (EfficientNet-B0)
        ↓
  GlobalAveragePooling
        ↓
  Concatenate with gender feature (binary)
        ↓
  FC(512) → ReLU → Dropout(0.3) → FC(1)
        ↓
  Denormalize output (× σ_age ≈ 41 months)
```

### Ensemble Inference
Seven models (VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNet-B0, EfficientNet-B3, Xception) each produce an independent prediction. The ensemble mean is the final estimate; the standard deviation provides the confidence interval.

```
ŷ_ensemble = (1/7) × Σ ŷ_k
CI_95      = ŷ_ensemble ± 1.96 × σ(ŷ_1 ... ŷ_7)
```

---

## Training

All models were trained under identical hyperparameters on the RSNA 2017 dataset:

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Max epochs | 40 |
| Early stopping | patience = 8 |
| Loss | MSE |
| Random seed | 42 |

Training was performed on Kaggle (GPU T4 × 2). To reproduce:

1. Upload `final-final.ipynb` to Kaggle
2. Attach the [RSNA Bone Age dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age)
3. Run all cells — model checkpoints are saved per epoch

---

## Clinical Output

The generated report maps bone age to clinical context:

| Bone Age | Growth Stage | Orthodontic Implication |
|----------|-------------|------------------------|
| < 24 mo | Infancy | Pre-treatment monitoring |
| 24–60 mo | Early Childhood | Interceptive screening |
| 60–120 mo | Mid-Childhood | Early orthopedic planning |
| 120–156 mo | Early Adolescence | Peak growth — optimal appliance window |
| 156–192 mo | Mid-Adolescence | Functional appliance last chance |
| > 192 mo | Late Adolescence | Post-growth; surgical planning |

---

## Paper

The IEEE-format paper (`report_ieee.tex`) describes the full system design, evaluation methodology, and clinical application. Compile locally with:

```bash
pdflatex report_ieee.tex
bibtex report_ieee
pdflatex report_ieee.tex
pdflatex report_ieee.tex
```

Or upload to [Overleaf](https://www.overleaf.com) and compile directly.
