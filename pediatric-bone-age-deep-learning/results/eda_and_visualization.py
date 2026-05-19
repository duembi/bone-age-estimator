"""
RSNA Pediatric Bone Age — EDA & Visualization Suite
Run this script after training is complete.
All plots are saved to /kaggle/working/plots/.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B3_Weights,
    VGG16_Weights, ResNet50_Weights, Inception_V3_Weights,
    DenseNet121_Weights,
)
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ─────────────────────────── Paths & Config ────────────────────
BASE_INPUT = Path("/kaggle/input/datasets/kmader/rsna-bone-age")
IMG_DIR    = BASE_INPUT / "boneage-training-dataset" / "boneage-training-dataset"
CSV_PATH   = BASE_INPUT / "boneage-training-dataset.csv"
OUTPUT_DIR = Path("/kaggle/working")
PLOT_DIR   = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Model priority order: use the best available checkpoint
MODEL_PRIORITY = [
    "EfficientNetB3", "EfficientNetB0", "DenseNet121",
    "ResNet50", "InceptionV3", "VGG16",
]
MODEL_IMG_SIZE = {
    "VGG16": 256, "ResNet50": 256, "InceptionV3": 299,
    "DenseNet121": 256, "EfficientNetB0": 256, "EfficientNetB3": 300,
}

SEED      = 42
VAL_SPLIT = 0.18
IMG_SIZE  = 256   # default; overridden per model
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = {"male": "#4C72B0", "female": "#DD8452", "train": "#2ecc71", "val": "#e74c3c"}

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def save(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def apply_clahe(img):
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


def load_img(img_path, size=IMG_SIZE):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = apply_clahe(img)
    img = pad_to_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# ══════════════════════════════════════════════════════════════
#  1. EDA — METADATA
# ══════════════════════════════════════════════════════════════

def eda_metadata(df):
    print("\n[1/7] EDA — Metadata plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("RSNA Bone Age — Metadata EDA", fontsize=16, fontweight="bold")

    # 1a. Age distribution — histogram + KDE
    ax = axes[0, 0]
    sns.histplot(df["boneage"], bins=40, kde=True, color="#5B84B1", ax=ax)
    mean_val   = df["boneage"].mean()
    median_val = df["boneage"].median()
    ax.axvline(mean_val,   color="red",    ls="--", lw=1.5, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="orange", ls="--", lw=1.5, label=f"Median: {median_val:.1f}")
    ax.set_title("Bone Age Distribution (months)")
    ax.set_xlabel("Bone Age (months)")
    ax.legend()

    # 1b. Gender distribution
    ax = axes[0, 1]
    counts = df["male"].value_counts()
    labels = ["Male" if v else "Female" for v in counts.index]
    colors = [COLORS["male"], COLORS["female"]]
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(12)
    ax.set_title("Gender Distribution")

    # 1c. Age distribution by gender — violin
    ax = axes[0, 2]
    df_plot = df.copy()
    df_plot["Gender"] = df_plot["male"].map({True: "Male", False: "Female"})
    sns.violinplot(data=df_plot, x="Gender", y="boneage",
                   palette={"Male": COLORS["male"], "Female": COLORS["female"]},
                   inner="box", ax=ax)
    ax.set_title("Bone Age by Gender")
    ax.set_ylabel("Bone Age (months)")

    # 1d. Sample count by age group
    ax = axes[1, 0]
    df_plot2 = df.copy()
    df_plot2["Age Group"] = pd.cut(
        df_plot2["boneage"],
        bins=[0, 24, 60, 120, 180, 228],
        labels=["0-2y", "2-5y", "5-10y", "10-15y", "15-19y"]
    )
    group_counts = df_plot2.groupby(["Age Group", df_plot2["male"].map({True:"Male", False:"Female"})]).size().unstack()
    group_counts.plot(kind="bar", ax=ax, color=[COLORS["female"], COLORS["male"]], edgecolor="white")
    ax.set_title("Sample Count by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Gender")

    # 1e. Box plot — by gender
    ax = axes[1, 1]
    sns.boxplot(data=df_plot, x="Gender", y="boneage",
                palette={"Male": COLORS["male"], "Female": COLORS["female"]},
                width=0.4, ax=ax)
    ax.set_title("Box Plot — Bone Age")
    ax.set_ylabel("Bone Age (months)")

    # 1f. Summary statistics table
    ax = axes[1, 2]
    ax.axis("off")
    stats_df = df.groupby(df["male"].map({True:"Male", False:"Female"}))["boneage"].describe().round(1)
    table = ax.table(
        cellText=stats_df.values,
        rowLabels=stats_df.index,
        colLabels=stats_df.columns,
        cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    ax.set_title("Summary Statistics", pad=20)

    plt.tight_layout()
    save(fig, "01_eda_metadata.png")


# ══════════════════════════════════════════════════════════════
#  2. EDA — IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════

def eda_images(df, n_samples=12):
    print("[2/7] EDA — Sample images (before/after CLAHE)...")

    sample_rows = df.sample(n=6, random_state=SEED)

    fig, axes = plt.subplots(4, 6, figsize=(20, 14))
    fig.suptitle("Sample Radiographs: Raw vs CLAHE Processed", fontsize=14, fontweight="bold")

    for col_idx, (_, row) in enumerate(sample_rows.iterrows()):
        img_path = str(IMG_DIR / f"{int(row['id'])}.png")
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        # Row 0: Raw image
        raw_sq = pad_to_square(raw)
        raw_rs = cv2.resize(raw_sq, (IMG_SIZE, IMG_SIZE))
        axes[0, col_idx].imshow(raw_rs, cmap="bone")
        axes[0, col_idx].set_title(
            f"{'M' if row['male'] else 'F'} | {int(row['boneage'])}mo",
            fontsize=9
        )
        axes[0, col_idx].axis("off")

        # Row 1: Histogram (raw)
        axes[1, col_idx].hist(raw_rs.ravel(), bins=50, color="#5B84B1", alpha=0.8)
        axes[1, col_idx].set_xlim(0, 255)
        axes[1, col_idx].set_yticks([])
        axes[1, col_idx].set_xlabel("Pixel Value", fontsize=7)

        # Row 2: CLAHE applied
        clahe_img = apply_clahe(raw)
        clahe_sq  = pad_to_square(clahe_img)
        clahe_rs  = cv2.resize(clahe_sq, (IMG_SIZE, IMG_SIZE))
        axes[2, col_idx].imshow(clahe_rs, cmap="bone")
        axes[2, col_idx].set_title("CLAHE", fontsize=9)
        axes[2, col_idx].axis("off")

        # Row 3: Histogram (CLAHE)
        axes[3, col_idx].hist(clahe_rs.ravel(), bins=50, color="#DD8452", alpha=0.8)
        axes[3, col_idx].set_xlim(0, 255)
        axes[3, col_idx].set_yticks([])
        axes[3, col_idx].set_xlabel("Pixel Value", fontsize=7)

    # Row labels
    for row_i, label in enumerate(["Raw Image", "Raw Histogram", "CLAHE Image", "CLAHE Histogram"]):
        axes[row_i, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=10)

    plt.tight_layout()
    save(fig, "02_eda_images_clahe.png")


def eda_image_stats(df, n=500):
    print("[3/7] EDA — Image size & brightness statistics...")

    sample = df.sample(min(n, len(df)), random_state=SEED)
    heights, widths, means, stds = [], [], [], []

    for _, row in sample.iterrows():
        img = cv2.imread(str(IMG_DIR / f"{int(row['id'])}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        heights.append(img.shape[0])
        widths.append(img.shape[1])
        means.append(img.mean())
        stds.append(img.std())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Image Statistics (n=500 samples)", fontsize=13, fontweight="bold")

    # Size scatter
    axes[0].scatter(widths, heights, alpha=0.4, s=15, color="#5B84B1")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Image Size Distribution")
    axes[0].axline((0,0), slope=1, color="red", ls="--", lw=1, label="Square")
    axes[0].legend()

    # Brightness distribution
    axes[1].hist(means, bins=30, color="#2ecc71", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Mean Pixel Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Image Brightness Distribution")

    # Contrast distribution
    axes[2].hist(stds, bins=30, color="#e74c3c", edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("Pixel Std Deviation")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Image Contrast Distribution")

    plt.tight_layout()
    save(fig, "03_eda_image_stats.png")


# ══════════════════════════════════════════════════════════════
#  3. TRAINING CURVES
# ══════════════════════════════════════════════════════════════

def plot_training_curves(history_path=None):
    print("[4/7] Training curves...")

    # If history_path not provided, find the best available model's history
    if history_path is None or not Path(history_path).exists():
        model_name, _ = _find_best_checkpoint()
        if model_name:
            history_path = OUTPUT_DIR / f"history_{model_name}.csv"
        # fallback: legacy single-model history
        if history_path is None or not Path(history_path).exists():
            history_path = OUTPUT_DIR / "training_history.csv"

    if not Path(history_path).exists():
        print("  training_history.csv not found, skipping.")
        return

    hist = pd.read_csv(history_path)
    epochs = range(1, len(hist) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], color=COLORS["train"], lw=2, label="Train Loss")
    ax.plot(epochs, hist["val_loss"],   color=COLORS["val"],   lw=2, label="Val Loss")
    best_ep = hist["val_loss"].idxmin() + 1
    ax.axvline(best_ep, color="gray", ls="--", lw=1, label=f"Best epoch: {best_ep}")
    ax.fill_between(epochs, hist["train_loss"], hist["val_loss"], alpha=0.08, color="purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend()

    # MAE
    ax = axes[1]
    ax.plot(epochs, hist["train_mae"], color=COLORS["train"], lw=2, label="Train MAE")
    ax.plot(epochs, hist["val_mae"],   color=COLORS["val"],   lw=2, label="Val MAE")
    best_mae = hist["val_mae"].min()
    best_ep2 = hist["val_mae"].idxmin() + 1
    ax.axvline(best_ep2, color="gray", ls="--", lw=1, label=f"Best: {best_mae:.2f}m (ep {best_ep2})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (months)")
    ax.set_title("Train vs Validation MAE")
    ax.legend()

    plt.tight_layout()
    save(fig, "04_training_curves.png")


# ══════════════════════════════════════════════════════════════
#  4. MODEL PREDICTION ANALYSIS
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  MODEL + INFERENCE (compatible with multi-model system)
# ══════════════════════════════════════════════════════════════

class _InceptionV3Features(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b

    def forward(self, x):
        x = self.b.Conv2d_1a_3x3(x); x = self.b.Conv2d_2a_3x3(x)
        x = self.b.Conv2d_2b_3x3(x); x = self.b.maxpool1(x)
        x = self.b.Conv2d_3b_1x1(x); x = self.b.Conv2d_4a_3x3(x)
        x = self.b.maxpool2(x)
        for block in [self.b.Mixed_5b, self.b.Mixed_5c, self.b.Mixed_5d,
                      self.b.Mixed_6a, self.b.Mixed_6b, self.b.Mixed_6c,
                      self.b.Mixed_6d, self.b.Mixed_6e,
                      self.b.Mixed_7a, self.b.Mixed_7b, self.b.Mixed_7c]:
            x = block(x)
        return x


class _DenseFeatures(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        return F.relu(self.features(x), inplace=True)


def _backbone(name):
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
        return _DenseFeatures(m.features), 1024
    elif name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        return m.features, 1280
    elif name == "EfficientNetB3":
        m = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        return m.features, 1536
    else:
        raise ValueError(name)


class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name, dropout=0.4):
        super().__init__()
        ib, fd = _backbone(backbone_name)
        self.image_branch  = ib
        self.gap           = nn.AdaptiveAvgPool2d(1)
        self.gender_branch = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fd + 32, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, img, gender):
        f = self.gap(self.image_branch(img)).flatten(1)
        return self.head(torch.cat([f, self.gender_branch(gender)], dim=1)).squeeze(1)


class _InferDataset(torch.utils.data.Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, df, img_dir, img_size=256):
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
        row      = self.df.iloc[idx]
        img_path = str(self.img_dir / f"{int(row['id'])}.png")
        img      = load_img(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.stack([img] * 3, axis=-1)
        return (
            self.tf(Image.fromarray(img)),
            torch.tensor([float(row["male"])], dtype=torch.float32),
            torch.tensor(float(row["boneage"]), dtype=torch.float32),
        )


def _find_best_checkpoint():
    """Return (model_name, ckpt_path) for the best available checkpoint."""
    for name in MODEL_PRIORITY:
        p = OUTPUT_DIR / f"best_{name}.pth"
        if p.exists():
            return name, p
    # fallback: legacy single-model checkpoint
    legacy = OUTPUT_DIR / "best_model.pth"
    if legacy.exists():
        return "EfficientNetB0", legacy
    return None, None


def get_predictions(df_val):
    model_name, ckpt_path = _find_best_checkpoint()
    if ckpt_path is None:
        print("  Checkpoint not found, skipping prediction plots.")
        return None, None, None

    print(f"  Using model for inference: {model_name}")
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    age_scale = ckpt.get("age_scale", 1.0)
    img_size  = MODEL_IMG_SIZE.get(model_name, 256)

    model = BoneAgeModel(model_name).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds     = _InferDataset(df_val, IMG_DIR, img_size=img_size)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    preds, trues, genders = [], [], []
    with torch.no_grad():
        for img, gender, label in loader:
            out = model(img.to(DEVICE), gender.to(DEVICE))
            preds.extend((out.cpu().numpy() * age_scale).tolist())
            trues.extend(label.numpy().tolist())
            genders.extend(gender.squeeze(1).numpy().tolist())

    return np.array(preds), np.array(trues), np.array(genders)


def plot_prediction_analysis(df_val):
    print("[5/7] Prediction analysis plots...")

    preds, trues, genders = get_predictions(df_val)
    if preds is None:
        return

    errors = preds - trues

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Model Prediction Analysis (Validation Set)", fontsize=15, fontweight="bold")

    gender_labels = np.where(genders == 1.0, "Male", "Female")
    palette = {"Male": COLORS["male"], "Female": COLORS["female"]}

    # 5a. Predicted vs True scatter
    ax = axes[0, 0]
    for g, c in palette.items():
        mask = gender_labels == g
        ax.scatter(trues[mask], preds[mask], alpha=0.35, s=12, color=c, label=g)
    mn, mx = trues.min(), trues.max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True Bone Age (months)")
    ax.set_ylabel("Predicted Age (months)")
    ax.set_title("Predicted vs True")
    ax.legend(markerscale=2)

    mae  = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    ax.text(0.05, 0.95, f"MAE: {mae:.2f}m\nRMSE: {rmse:.2f}m",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # 5b. Residual distribution
    ax = axes[0, 1]
    sns.histplot(errors, bins=50, kde=True, color="#8E44AD", ax=ax)
    ax.axvline(0, color="red", ls="--", lw=1.5, label="Zero error")
    ax.set_xlabel("Error (Predicted - True, months)")
    bias_val = errors.mean()
    ax.set_title(f"Residual Distribution  |  Bias: {bias_val:.2f}m")
    ax.legend()

    # 5c. Residual vs Predicted (heteroscedasticity check)
    ax = axes[0, 2]
    for g, c in palette.items():
        mask = gender_labels == g
        ax.scatter(preds[mask], errors[mask], alpha=0.3, s=12, color=c, label=g)
    ax.axhline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Predicted (months)")
    ax.set_ylabel("Residual Error (months)")
    ax.set_title("Residual vs Predicted (Heteroscedasticity)")
    ax.legend(markerscale=2)

    # 5d. Bland-Altman plot
    ax = axes[1, 0]
    mean_vals = (preds + trues) / 2
    ax.scatter(mean_vals, errors, alpha=0.3, s=12, color="#16A085")
    mean_err = errors.mean()
    std_err  = errors.std()
    lo_lim   = mean_err - 1.96 * std_err
    hi_lim   = mean_err + 1.96 * std_err
    ax.axhline(mean_err, color="red",  ls="-",  lw=1.5, label=f"Mean: {mean_err:.2f}")
    ax.axhline(hi_lim,   color="blue", ls="--", lw=1,   label=f"+1.96σ: {hi_lim:.2f}")
    ax.axhline(lo_lim,   color="blue", ls="--", lw=1,   label=f"-1.96σ: {lo_lim:.2f}")
    ax.set_xlabel("Mean (months)")
    ax.set_ylabel("Difference (months)")
    ax.set_title("Bland-Altman Analysis")
    ax.legend(fontsize=8)

    # 5e. MAE by age group
    ax = axes[1, 1]
    bins   = [0, 24, 60, 120, 180, 228]
    labels = ["0-2y", "2-5y", "5-10y", "10-15y", "15-19y"]
    bucket = pd.cut(trues, bins=bins, labels=labels)
    mae_by_group = pd.Series(np.abs(errors)).groupby(bucket).mean()
    colors_bar = sns.color_palette("Blues_d", len(mae_by_group))
    mae_by_group.plot(kind="bar", ax=ax, color=colors_bar, edgecolor="white")
    ax.set_title("MAE by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("MAE (months)")
    ax.tick_params(axis="x", rotation=0)

    # 5f. Q-Q plot (normality test)
    ax = axes[1, 2]
    (osm, osr), (slope, intercept, r) = stats.probplot(errors, dist="norm")
    ax.scatter(osm, osr, alpha=0.4, s=12, color="#E67E22")
    ax.plot(osm, slope*np.array(osm)+intercept, "r-", lw=1.5, label=f"R²={r**2:.3f}")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot (Residual Normality)")
    ax.legend()

    plt.tight_layout()
    save(fig, "05_prediction_analysis.png")
    return preds, trues, genders


# ══════════════════════════════════════════════════════════════
#  5. CONFUSION MATRIX (Age Groups)
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(preds, trues):
    print("[6/7] Confusion matrix (age groups)...")

    bins   = [0, 24, 60, 120, 180, 228]
    labels = ["0-2y", "2-5y", "5-10y", "10-15y", "15-19y"]

    true_cls = pd.cut(trues, bins=bins, labels=labels)
    pred_cls = pd.cut(np.clip(preds, 0, 228), bins=bins, labels=labels)

    # Remove NaN values
    mask = (~true_cls.isna()) & (~pred_cls.isna())
    true_cls = true_cls[mask]
    pred_cls = pred_cls[mask]

    cm = confusion_matrix(true_cls, pred_cls, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — Age Groups (Regression → Classification)", fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                linewidths=0.5, cbar_kws={"label": "Sample Count"})
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Raw Counts")

    # Normalized (recall)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                linewidths=0.5, vmin=0, vmax=1, cbar_kws={"label": "Recall Ratio"})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Normalized (Row = Recall)")

    plt.tight_layout()
    save(fig, "06_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════
#  6. SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════

def plot_summary_dashboard(df, preds, trues, history_path):
    print("[7/7] Summary dashboard...")

    hist = pd.read_csv(history_path) if history_path.exists() else None

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle("RSNA Bone Age — Project Summary Dashboard", fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Age distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df["boneage"], bins=30, kde=True, color="#5B84B1", ax=ax1)
    ax1.set_title("Age Distribution")
    ax1.set_xlabel("Month")

    # Gender
    ax2 = fig.add_subplot(gs[0, 1])
    counts = df["male"].value_counts()
    ax2.pie(counts, labels=["Male", "Female"], colors=[COLORS["male"], COLORS["female"]],
            autopct="%1.1f%%", startangle=90)
    ax2.set_title("Gender")

    # Train/Val Loss
    ax3 = fig.add_subplot(gs[0, 2])
    if hist is not None:
        ax3.plot(hist["train_loss"], color=COLORS["train"], label="Train")
        ax3.plot(hist["val_loss"],   color=COLORS["val"],   label="Val")
        ax3.set_title("Loss")
        ax3.legend(fontsize=8)
        ax3.set_xlabel("Epoch")

    # Train/Val MAE
    ax4 = fig.add_subplot(gs[0, 3])
    if hist is not None:
        ax4.plot(hist["train_mae"], color=COLORS["train"], label="Train")
        ax4.plot(hist["val_mae"],   color=COLORS["val"],   label="Val")
        ax4.set_title("MAE (months)")
        ax4.legend(fontsize=8)
        ax4.set_xlabel("Epoch")

    # Predicted vs True
    ax5 = fig.add_subplot(gs[1, :2])
    if preds is not None:
        ax5.scatter(trues, preds, alpha=0.3, s=8, color="#8E44AD")
        mn, mx = trues.min(), trues.max()
        ax5.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        mae = np.abs(preds - trues).mean()
        ax5.set_title(f"Predicted vs True  |  MAE: {mae:.2f} months")
        ax5.set_xlabel("True (months)")
        ax5.set_ylabel("Predicted (months)")

    # Error distribution
    ax6 = fig.add_subplot(gs[1, 2:])
    if preds is not None:
        errors = preds - trues
        sns.histplot(errors, bins=40, kde=True, color="#E67E22", ax=ax6)
        ax6.axvline(0, color="red", ls="--", lw=1.5)
        ax6.set_title("Error Distribution")
        ax6.set_xlabel("Error (months)")

    # Summary table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    if preds is not None and hist is not None:
        best_mae = hist["val_mae"].min()
        best_ep  = hist["val_mae"].idxmin() + 1
        total_ep = len(hist)
        rmse     = np.sqrt(((preds - trues)**2).mean())
        bias     = (preds - trues).mean()
        summary = [
            ["Total Samples",  f"{len(df):,}"],
            ["Train / Val",    f"{int(len(df)*(1-VAL_SPLIT)):,} / {int(len(df)*VAL_SPLIT):,}"],
            ["Best Epoch",     f"{best_ep} / {total_ep}"],
            ["Best Val MAE",   f"{best_mae:.2f} months"],
            ["RMSE",           f"{rmse:.2f} months"],
            ["Bias",           f"{bias:.2f} months"],
        ]
        table = ax7.table(
            cellText=[[r[1] for r in summary]],
            colLabels=[r[0] for r in summary],
            cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)
        ax7.set_title("Summary Metrics", pad=15)

    save(fig, "00_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"Plots → {PLOT_DIR}\n")

    df = pd.read_csv(CSV_PATH)
    df["male"] = df["male"].astype(bool)

    df["age_bucket"] = pd.cut(df["boneage"], bins=10, labels=False)
    _, val_df = train_test_split(df, test_size=VAL_SPLIT, stratify=df["age_bucket"], random_state=SEED)

    # EDA
    eda_metadata(df)
    eda_images(df)
    eda_image_stats(df)

    # Training curves (auto-finds the best checkpoint's history)
    plot_training_curves()

    # Prediction analysis
    result = plot_prediction_analysis(val_df)
    preds, trues = (result[0], result[1]) if result is not None else (None, None)

    # Confusion matrix
    if preds is not None:
        plot_confusion_matrix(preds, trues)

    # Summary dashboard
    model_name, _ = _find_best_checkpoint()
    best_hist = OUTPUT_DIR / f"history_{model_name}.csv" if model_name else OUTPUT_DIR / "training_history.csv"
    plot_summary_dashboard(df, preds, trues, best_hist)

    print(f"\nAll plots saved → {PLOT_DIR}")
    print("Files:")
    for f in sorted(PLOT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
