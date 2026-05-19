"""
Automated Pediatric Bone Age Estimation System
Comprehensive Evaluation Suite — RSNA Bone Age Challenge 2017

Evaluates all trained models across 21 metrics in 3 categories:
  1. REGRESSION    : MAE, MSE, RMSE, MAPE, Median AE, R², Explained Variance
  2. mAP (adapted) : AP@3m, AP@6m, AP@12m, AP@24m, mAP
  3. CLASSIFICATION: Accuracy, Precision (macro/weighted), Recall (macro/weighted),
                     F1 (macro/weighted), Cohen's Kappa

Run after multi_model_train.py. Outputs: all_models_metrics.csv
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import (
    VGG16_Weights, ResNet50_Weights, Inception_V3_Weights,
    DenseNet121_Weights, EfficientNet_B0_Weights, EfficientNet_B3_Weights,
)
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, explained_variance_score,
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, classification_report,
    confusion_matrix,
)

# ─────────────────────────── Paths ─────────────────────────────
BASE_INPUT = Path("/kaggle/input/datasets/kmader/rsna-bone-age")
IMG_DIR    = BASE_INPUT / "boneage-training-dataset" / "boneage-training-dataset"
CSV_PATH   = BASE_INPUT / "boneage-training-dataset.csv"
OUTPUT_DIR = Path("/kaggle/working")
PLOT_DIR   = OUTPUT_DIR / "eval_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEED      = 42
VAL_SPLIT = 0.18
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)

# Age bucket labels for classification metrics
AGE_BINS   = [0, 24, 60, 120, 180, 228]
AGE_LABELS = ["0-2y", "2-5y", "5-10y", "10-15y", "15-19y"]

# Models registered in multi_model_train.py
MODEL_REGISTRY = {
    "VGG16":          {"img_size": 256, "feature_dim": 512},
    "ResNet50":       {"img_size": 256, "feature_dim": 2048},
    "InceptionV3":    {"img_size": 299, "feature_dim": 2048},
    "DenseNet121":    {"img_size": 256, "feature_dim": 1024},
    "EfficientNetB0": {"img_size": 256, "feature_dim": 1280},
    "EfficientNetB3": {"img_size": 300, "feature_dim": 1536},
    "Xception":       {"img_size": 299, "feature_dim": 2048},
}


# ══════════════════════════════════════════════════════════════
#  1. MODEL DEFINITIONS (same as multi_model_train.py)
# ══════════════════════════════════════════════════════════════

def apply_clahe(img):
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def pad_to_square(img):
    h, w = img.shape
    if h == w:
        return img
    size = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = img
    return canvas


def load_img(img_path, img_size):
    import cv2
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img = apply_clahe(img)
    img = pad_to_square(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return np.stack([img, img, img], axis=-1)


class _InferDataset(torch.utils.data.Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, df, img_dir, img_size):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.img_size = img_size
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = load_img(self.img_dir / f"{int(row['id'])}.png", self.img_size)
        return (
            self.tf(Image.fromarray(img)),
            torch.tensor([float(row["male"])], dtype=torch.float32),
            torch.tensor(float(row["boneage"]), dtype=torch.float32),
        )


class _InceptionV3Features(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.b = backbone

    def forward(self, x):
        x = self.b.Conv2d_1a_3x3(x);  x = self.b.Conv2d_2a_3x3(x)
        x = self.b.Conv2d_2b_3x3(x);  x = self.b.maxpool1(x)
        x = self.b.Conv2d_3b_1x1(x);  x = self.b.Conv2d_4a_3x3(x)
        x = self.b.maxpool2(x)
        x = self.b.Mixed_5b(x);  x = self.b.Mixed_5c(x);  x = self.b.Mixed_5d(x)
        x = self.b.Mixed_6a(x);  x = self.b.Mixed_6b(x);  x = self.b.Mixed_6c(x)
        x = self.b.Mixed_6d(x);  x = self.b.Mixed_6e(x)
        x = self.b.Mixed_7a(x);  x = self.b.Mixed_7b(x);  x = self.b.Mixed_7c(x)
        return x


class _DenseNet121Features(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        return F.relu(self.features(x), inplace=True)


def _build_backbone(name):
    if name == "VGG16":
        m = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        return m.features, 512
    elif name == "ResNet50":
        m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(m.children())[:-2]), 2048
    elif name == "InceptionV3":
        m = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        m.aux_logits = False
        return _InceptionV3Features(m), 2048
    elif name == "DenseNet121":
        m = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        return _DenseNet121Features(m.features), 1024
    elif name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        return m.features, 1280
    elif name == "EfficientNetB3":
        m = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        return m.features, 1536
    elif name == "Xception":
        import timm
        m = timm.create_model("xception", pretrained=False, num_classes=0, global_pool="")
        return m, 2048
    else:
        raise ValueError(f"Unknown: {name}")


class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name, dropout=0.4):
        super().__init__()
        self.backbone_name = backbone_name
        image_branch, feature_dim = _build_backbone(backbone_name)
        self.image_branch = image_branch
        self.gap           = nn.AdaptiveAvgPool2d(1)
        self.gender_branch = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim + 32, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, img, gender):
        feat   = self.gap(self.image_branch(img)).flatten(1)
        g_feat = self.gender_branch(gender)
        return self.head(torch.cat([feat, g_feat], dim=1)).squeeze(1)


# ══════════════════════════════════════════════════════════════
#  2. INFERENCE — Get predictions from a saved checkpoint
# ══════════════════════════════════════════════════════════════

def get_predictions(model_name: str, val_df: pd.DataFrame):
    ckpt_path = OUTPUT_DIR / f"best_{model_name}.pth"
    if not ckpt_path.exists():
        print(f"  [{model_name}] Checkpoint not found: {ckpt_path}")
        return None, None, None

    ckpt     = torch.load(ckpt_path, map_location=DEVICE)
    age_scale = ckpt.get("age_scale", 1.0)

    model = BoneAgeModel(model_name).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_size = MODEL_REGISTRY[model_name]["img_size"]
    ds       = _InferDataset(val_df, IMG_DIR, img_size)
    loader   = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    preds, trues, genders = [], [], []
    with torch.no_grad():
        for img, gender, label in loader:
            out = model(img.to(DEVICE), gender.to(DEVICE))
            # Denormalize predictions to original months
            preds.extend((out.cpu().numpy() * age_scale).tolist())
            trues.extend(label.numpy().tolist())
            genders.extend(gender.squeeze(1).numpy().tolist())

    return np.array(preds), np.array(trues), np.array(genders)


# ══════════════════════════════════════════════════════════════
#  3. METRIC SUITE
# ══════════════════════════════════════════════════════════════

def adapted_map(y_true: np.ndarray, y_pred: np.ndarray,
                thresholds=(3, 6, 12, 24)) -> float:
    """
    Adapted mAP for regression.
    At each error threshold t, compute precision = fraction of samples
    where |pred - true| <= t.  Average across all thresholds.
    """
    return float(np.mean([np.mean(np.abs(y_pred - y_true) <= t)
                          for t in thresholds]))


def age_to_bucket(ages: np.ndarray) -> np.ndarray:
    return pd.cut(np.clip(ages, 0, 228), bins=AGE_BINS,
                  labels=AGE_LABELS).astype(str)


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        model_name: str = "") -> dict:
    """
    Three metric categories:

    1. REGRESSION  — MAE, MSE, RMSE, MAPE, Median AE, R², Explained Variance
    2. mAP         — Adapted mAP at thresholds [3, 6, 12, 24] months
                     (AP@3m, AP@6m, AP@12m, AP@24m, mAP)
    3. CLASSIFICATION (age buckets: 0-2y, 2-5y, 5-10y, 10-15y, 15-19y)
                   — Accuracy, Precision macro/weighted,
                     Recall macro/weighted, F1 macro/weighted, Cohen's Kappa
    """
    errors    = y_pred - y_true
    safe_true = np.where(np.abs(y_true) < 1e-6, 1.0, y_true)

    # ── 1. Regression ────────────────────────────────────────────
    mae    = mean_absolute_error(y_true, y_pred)
    mse    = mean_squared_error(y_true, y_pred)
    rmse   = float(np.sqrt(mse))
    mape   = float(np.mean(np.abs(errors / safe_true)) * 100)
    med_ae = median_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    ev     = explained_variance_score(y_true, y_pred)

    # ── 2. mAP ───────────────────────────────────────────────────
    thresholds = (3, 6, 12, 24)
    ap_per_t   = {f"AP@{t}m": round(float(np.mean(np.abs(errors) <= t)), 4)
                  for t in thresholds}
    map_score  = adapted_map(y_true, y_pred, thresholds)

    # ── 3. Classification (age buckets) ──────────────────────────
    true_cls = age_to_bucket(y_true)
    pred_cls = age_to_bucket(y_pred)
    mask     = (true_cls != "nan") & (pred_cls != "nan")
    tc, pc   = true_cls[mask], pred_cls[mask]

    acc      = accuracy_score(tc, pc)
    prec_mac = precision_score(tc, pc, average="macro",    zero_division=0, labels=AGE_LABELS)
    rec_mac  = recall_score(  tc, pc, average="macro",    zero_division=0, labels=AGE_LABELS)
    f1_mac   = f1_score(      tc, pc, average="macro",    zero_division=0, labels=AGE_LABELS)
    prec_wt  = precision_score(tc, pc, average="weighted", zero_division=0, labels=AGE_LABELS)
    rec_wt   = recall_score(  tc, pc, average="weighted", zero_division=0, labels=AGE_LABELS)
    f1_wt    = f1_score(      tc, pc, average="weighted", zero_division=0, labels=AGE_LABELS)
    kappa    = cohen_kappa_score(tc, pc, labels=AGE_LABELS)

    metrics = {
        # ── Regression ──
        "MAE (months)":       round(mae,    3),
        "MSE":                round(mse,    3),
        "RMSE (months)":      round(rmse,   3),
        "MAPE (%)":           round(mape,   3),
        "Median AE (months)": round(med_ae, 3),
        "R²":                 round(r2,     4),
        "Explained Variance": round(ev,     4),
        # ── mAP ──
        **ap_per_t,
        "mAP @[3,6,12,24]m": round(map_score, 4),
        # ── Classification ──
        "Accuracy":           round(acc,      4),
        "Precision Macro":    round(prec_mac, 4),
        "Recall Macro":       round(rec_mac,  4),
        "F1 Macro":           round(f1_mac,   4),
        "Precision Weighted": round(prec_wt,  4),
        "Recall Weighted":    round(rec_wt,   4),
        "F1 Weighted":        round(f1_wt,    4),
        "Cohen's Kappa":      round(kappa,    4),
    }

    if model_name:
        metrics["Model"] = model_name

    return metrics


# ══════════════════════════════════════════════════════════════
#  4. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def plot_model_comparison(all_metrics: list):
    """Bar chart comparison across all three metric categories."""
    df_m = pd.DataFrame(all_metrics).set_index("Model")

    # 2 rows × 3 cols: one panel per key metric
    key_metrics = [
        ("MAE (months)",        "Regression",       True),   # lower better
        ("RMSE (months)",       "Regression",       True),
        ("R²",                  "Regression",       False),  # higher better
        ("mAP @[3,6,12,24]m",  "mAP",              False),
        ("F1 Weighted",         "Classification",   False),
        ("Cohen's Kappa",       "Classification",   False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Model Comparison — Key Metrics", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    palette = sns.color_palette("tab10", n_colors=len(df_m))

    for i, (metric, category, lower_better) in enumerate(key_metrics):
        if metric not in df_m.columns:
            continue
        ax   = axes[i]
        vals = df_m[metric]
        bars = ax.bar(df_m.index, vals, color=palette, edgecolor="white")
        ax.set_title(f"[{category}] {metric}", fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002 * (vals.max() or 1),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        best_idx = vals.idxmin() if lower_better else vals.idxmax()
        idx_pos  = list(df_m.index).index(best_idx)
        bars[idx_pos].set_edgecolor("#27AE60")
        bars[idx_pos].set_linewidth(2.5)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "07_model_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → 07_model_comparison_bars.png")


def plot_radar_chart(all_metrics: list):
    """Radar (spider) chart of normalized metrics per model — 3 categories."""
    df_m = pd.DataFrame(all_metrics).set_index("Model")

    # One axis per category: MAE (inv), R², mAP, Accuracy, F1 Weighted, Kappa
    def norm_score(model_row, metric, lower_better):
        col_vals = df_m[metric]
        if lower_better:
            return 1.0 - (model_row[metric] - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)
        return (model_row[metric] - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)

    axes_def = [
        ("MAE (months)",       True,  "MAE\n(inv)"),
        ("R²",                 False, "R²"),
        ("mAP @[3,6,12,24]m", False, "mAP"),
        ("Accuracy",           False, "Accuracy"),
        ("F1 Weighted",        False, "F1\nWeighted"),
        ("Cohen's Kappa",      False, "Kappa"),
    ]

    scores = {}
    for model in df_m.index:
        row = df_m.loc[model]
        scores[model] = [norm_score(row, m, lb) for m, lb, _ in axes_def]

    labels = [lbl for _, _, lbl in axes_def]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Model Radar Comparison", fontsize=14, fontweight="bold")

    palette = sns.color_palette("tab10", n_colors=len(scores))
    for (model, vals), color in zip(scores.items(), palette):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, vals_plot, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))

    fig.savefig(PLOT_DIR / "08_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → 08_radar_chart.png")


def plot_per_model_scatter(model_results: dict):
    """Predicted vs actual scatter for every model in a single figure."""
    n_models = len(model_results)
    if n_models == 0:
        return
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    palette = sns.color_palette("tab10", n_colors=n_models)

    for ax, (name, (preds, trues, _)), color in zip(
            axes, model_results.items(), palette):
        mae  = np.abs(preds - trues).mean()
        r2   = r2_score(trues, preds)
        mn, mx = trues.min(), trues.max()
        ax.scatter(trues, preds, alpha=0.3, s=8, color=color)
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_title(f"{name}\nMAE={mae:.2f}m | R²={r2:.3f}", fontweight="bold")
        ax.set_xlabel("True Age (months)")
        ax.set_ylabel("Predicted (months)")

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle("All Models — Predicted vs True", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "09_all_models_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → 09_all_models_scatter.png")


def plot_metrics_heatmap(all_metrics: list):
    """Heatmap of all metrics (3 categories) normalized to [0,1]."""
    df_m = pd.DataFrame(all_metrics).set_index("Model")
    df_num = df_m.select_dtypes(include=[np.number]).copy()

    # Invert error metrics so green = good everywhere
    error_cols = ["MAE (months)", "MSE", "RMSE (months)", "MAPE (%)", "Median AE (months)"]
    for col in error_cols:
        if col in df_num.columns:
            df_num[col] = df_num[col].max() - df_num[col]

    df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(22, max(5, len(df_norm) * 1.3)))
    sns.heatmap(
        df_norm.T, annot=df_m.select_dtypes(include=[np.number]).T.round(3).astype(str),
        fmt="", cmap="RdYlGn", ax=ax,
        linewidths=0.4, cbar_kws={"label": "Normalized Score (↑ better)"},
    )
    ax.set_title(
        "Metric Heatmap — Regression | mAP | Classification\n"
        "(Error metrics are inverted: dark green = good)",
        fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "10_metrics_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → 10_metrics_heatmap.png")


def plot_age_group_mae(model_results: dict):
    """Per-age-group MAE breakdown for each model."""
    rows_data = []
    for model_name, (preds, trues, _) in model_results.items():
        errors = np.abs(preds - trues)
        bucket = age_to_bucket(trues)
        for label in AGE_LABELS:
            mask = bucket == label
            if mask.sum() == 0:
                continue
            rows_data.append({
                "Model":       model_name,
                "Age Group":   label,
                "MAE (months)": errors[mask].mean(),
            })
    df_ag = pd.DataFrame(rows_data)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_ag, x="Age Group", y="MAE (months)", hue="Model",
                palette="tab10", ax=ax)
    ax.set_title("MAE by Age Group — Model Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("MAE (months)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "11_age_group_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → 11_age_group_mae.png")


def plot_training_histories():
    """Overlay training curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training History — All Models", fontsize=14, fontweight="bold")
    palette = sns.color_palette("tab10")

    for i, model_name in enumerate(MODEL_REGISTRY.keys()):
        hist_path = OUTPUT_DIR / f"history_{model_name}.csv"
        if not hist_path.exists():
            continue
        hist   = pd.read_csv(hist_path)
        epochs = range(1, len(hist) + 1)
        color  = palette[i % len(palette)]
        axes[0].plot(epochs, hist["val_loss"], color=color, lw=1.5, label=model_name)
        axes[1].plot(epochs, hist["val_mae"],  color=color, lw=1.5, label=model_name)

    axes[0].set_title("Val Huber Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Val MAE (months)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "12_training_histories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → 12_training_histories.png")


def plot_gender_mae(model_results: dict):
    """MAE split by gender for each model."""
    rows_data = []
    for model_name, (preds, trues, genders) in model_results.items():
        errors = np.abs(preds - trues)
        for g_val, g_label in [(1.0, "Male"), (0.0, "Female")]:
            mask = genders == g_val
            if mask.sum() == 0:
                continue
            rows_data.append({
                "Model":        model_name,
                "Gender":       g_label,
                "MAE (months)": errors[mask].mean(),
            })
    df_g = pd.DataFrame(rows_data)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_g, x="Model", y="MAE (months)", hue="Gender",
                palette={"Male": "#4C72B0", "Female": "#DD8452"}, ax=ax)
    ax.set_title("MAE by Gender — Model Comparison",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "13_gender_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → 13_gender_mae.png")


# ══════════════════════════════════════════════════════════════
#  5. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"Evaluation plots → {PLOT_DIR}\n")

    df = pd.read_csv(CSV_PATH)
    df["male"] = df["male"].astype(bool)
    df["age_bucket"] = pd.cut(df["boneage"], bins=10, labels=False)
    _, val_df = train_test_split(df, test_size=VAL_SPLIT,
                                 stratify=df["age_bucket"], random_state=SEED)

    model_results = {}   # name → (preds, trues, genders)
    all_metrics   = []

    # ── Collect predictions from every trained model ──────────
    for model_name in MODEL_REGISTRY.keys():
        ckpt_path = OUTPUT_DIR / f"best_{model_name}.pth"
        if not ckpt_path.exists():
            print(f"  [{model_name}] No checkpoint found — skipping.")
            continue

        print(f"\n[{model_name}] Running inference...")
        preds, trues, genders = get_predictions(model_name, val_df)
        if preds is None:
            continue

        model_results[model_name] = (preds, trues, genders)

        # ── Compute all metrics ──
        print(f"  [{model_name}] Computing metrics...")
        metrics = compute_all_metrics(trues, preds, model_name=model_name)
        all_metrics.append(metrics)

        # Print key metrics from each category
        print(f"  [Regression]  MAE={metrics['MAE (months)']:.2f}  "
              f"RMSE={metrics['RMSE (months)']:.2f}  R²={metrics['R²']:.4f}")
        print(f"  [mAP]         AP@3m={metrics['AP@3m']:.3f}  AP@6m={metrics['AP@6m']:.3f}  "
              f"AP@12m={metrics['AP@12m']:.3f}  mAP={metrics['mAP @[3,6,12,24]m']:.4f}")
        kappa = metrics["Cohen's Kappa"]
        print(f"  [Classif.]    Acc={metrics['Accuracy']:.4f}  "
              f"F1w={metrics['F1 Weighted']:.4f}  Kappa={kappa:.4f}")

        # Full classification report
        true_cls = age_to_bucket(trues)
        pred_cls = age_to_bucket(preds)
        mask     = (true_cls != "nan") & (pred_cls != "nan")
        print(f"\n  [{model_name}] Classification Report:")
        print(classification_report(true_cls[mask], pred_cls[mask], labels=AGE_LABELS))

    if not all_metrics:
        print("No trained models found. Run multi_model_train.py first.")
        return

    # ── Save metrics to CSV ───────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "all_models_metrics.csv", index=False)
    print(f"\nMetrics saved → {OUTPUT_DIR / 'all_models_metrics.csv'}")

    # ── Ranking table ────────────────────────────────────────
    print("\n" + "="*70)
    print("RANKING BY MAE (lower is better)")
    print("="*70)
    rank_df = metrics_df[["Model",
                           "MAE (months)", "RMSE (months)", "R²",
                           "mAP @[3,6,12,24]m",
                           "Accuracy", "F1 Weighted", "Cohen's Kappa"]
                         ].sort_values("MAE (months)")
    print(rank_df.to_string(index=False))

    # ── Generate plots ────────────────────────────────────────
    print("\nGenerating plots...")
    plot_training_histories()
    plot_model_comparison(all_metrics)
    plot_radar_chart(all_metrics)
    plot_per_model_scatter(model_results)
    plot_metrics_heatmap(all_metrics)
    plot_age_group_mae(model_results)
    plot_gender_mae(model_results)

    print(f"\nAll evaluation plots → {PLOT_DIR}")


if __name__ == "__main__":
    main()
