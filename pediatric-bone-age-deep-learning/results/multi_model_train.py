"""
RSNA Pediatric Bone Age Challenge 2017
Multi-Model Comparison Training Pipeline
Models: VGG16, ResNet50, InceptionV3, DenseNet121, EfficientNet-B0, EfficientNet-B3
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    VGG16_Weights,
    ResNet50_Weights,
    Inception_V3_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
)
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("timm not found — Xception will be skipped. Install: pip install timm")

# ─────────────────────────── Paths ─────────────────────────────
BASE_INPUT = Path("/kaggle/input/datasets/kmader/rsna-bone-age")
IMG_DIR    = BASE_INPUT / "boneage-training-dataset" / "boneage-training-dataset"
CSV_PATH   = BASE_INPUT / "boneage-training-dataset.csv"
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Config ────────────────────────────
BATCH_SIZE   = 32
NUM_EPOCHS   = 40
LR           = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT    = 0.18
PATIENCE     = 8
NUM_WORKERS  = 4
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ──────────────── Model Registry ───────────────────────────────
# img_size: required input size, feature_dim: backbone output channels
MODEL_REGISTRY = {
    "VGG16":          {"img_size": 256, "feature_dim": 512},
    "ResNet50":       {"img_size": 256, "feature_dim": 2048},
    "InceptionV3":    {"img_size": 299, "feature_dim": 2048},
    "DenseNet121":    {"img_size": 256, "feature_dim": 1024},
    "EfficientNetB0": {"img_size": 256, "feature_dim": 1280},
    "EfficientNetB3": {"img_size": 300, "feature_dim": 1536},
    "Xception":       {"img_size": 299, "feature_dim": 2048},  # requires timm
}

# Models to train (modify this list to train a subset)
MODELS_TO_TRAIN = [m for m in MODEL_REGISTRY
                   if m != "Xception" or TIMM_AVAILABLE]

# ──────────────── Age Label Normalization ──────────────────────
# Inspired by top Kaggle notebooks: normalizing targets stabilizes training.
# We divide by 2*std (≈ 82 months) so labels fall in [0, ~2.8].
# All MAE values are reported in original months (denormalized).
AGE_SCALE: float = 1.0   # set in main() from training data std


# ══════════════════════════════════════════════════════════════
#  1. PREPROCESSING
# ══════════════════════════════════════════════════════════════

def apply_clahe(img_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    if h == w:
        return img
    size = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size - h) // 2:(size - h) // 2 + h,
           (size - w) // 2:(size - w) // 2 + w] = img
    return canvas


def load_and_preprocess(img_path: str, img_size: int) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img = apply_clahe(img)
    img = pad_to_square(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return np.stack([img, img, img], axis=-1)


# ══════════════════════════════════════════════════════════════
#  2. DATASET
# ══════════════════════════════════════════════════════════════

class BoneAgeDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, df: pd.DataFrame, img_dir: Path,
                 img_size: int = 256, augment: bool = False):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.img_size = img_size
        self.augment  = augment

        base = [transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD)]
        aug  = [transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)]
        self.transform = transforms.Compose((aug if augment else []) + base)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = str(self.img_dir / f"{int(row['id'])}.png")
        img_np   = load_and_preprocess(img_path, self.img_size)
        img_pil  = Image.fromarray(img_np)
        img_t    = self.transform(img_pil)
        gender   = torch.tensor([float(row["male"])], dtype=torch.float32)
        # Normalize target: divide by AGE_SCALE (set in main from 2*std)
        raw_age  = float(row["boneage"])
        label    = torch.tensor(raw_age / AGE_SCALE, dtype=torch.float32)
        return img_t, gender, label


# ══════════════════════════════════════════════════════════════
#  3. BACKBONE WRAPPERS
# ══════════════════════════════════════════════════════════════

class _InceptionV3Features(nn.Module):
    """Extract features from InceptionV3 up to Mixed_7c (2048 channels)."""
    def __init__(self, backbone):
        super().__init__()
        self.b = backbone

    def forward(self, x):
        x = self.b.Conv2d_1a_3x3(x)
        x = self.b.Conv2d_2a_3x3(x)
        x = self.b.Conv2d_2b_3x3(x)
        x = self.b.maxpool1(x)
        x = self.b.Conv2d_3b_1x1(x)
        x = self.b.Conv2d_4a_3x3(x)
        x = self.b.maxpool2(x)
        x = self.b.Mixed_5b(x)
        x = self.b.Mixed_5c(x)
        x = self.b.Mixed_5d(x)
        x = self.b.Mixed_6a(x)
        x = self.b.Mixed_6b(x)
        x = self.b.Mixed_6c(x)
        x = self.b.Mixed_6d(x)
        x = self.b.Mixed_6e(x)
        x = self.b.Mixed_7a(x)
        x = self.b.Mixed_7b(x)
        x = self.b.Mixed_7c(x)
        return x


class _DenseNet121Features(nn.Module):
    """DenseNet121 features + ReLU (ReLU not included in m.features)."""
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        return F.relu(self.features(x), inplace=True)


def build_backbone(name: str):
    """Return (feature_extractor_module, feature_dim)."""
    if name == "VGG16":
        m = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        return m.features, 512

    elif name == "ResNet50":
        m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove avgpool and fc
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
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Xception. pip install timm")
        # timm Xception: global_pool='' keeps spatial dims, num_classes=0 removes head
        m = timm.create_model("xception", pretrained=True, num_classes=0, global_pool="")
        return m, 2048

    else:
        raise ValueError(f"Unknown model: {name}")


# ══════════════════════════════════════════════════════════════
#  4. UNIFIED MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class BoneAgeModel(nn.Module):
    """
    Multi-input architecture for any backbone.
      Image branch  : pretrained backbone → GAP → feature_dim
      Gender branch : Linear(1→32) → ReLU
      Fusion        : Concat → FC head → scalar (regression)
    """

    def __init__(self, backbone_name: str, dropout: float = 0.4):
        super().__init__()
        self.backbone_name = backbone_name

        image_branch, feature_dim = build_backbone(backbone_name)
        self.image_branch = image_branch
        self.gap          = nn.AdaptiveAvgPool2d(1)

        self.gender_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim + 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, img: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        feat   = self.gap(self.image_branch(img)).flatten(1)
        g_feat = self.gender_branch(gender)
        return self.head(torch.cat([feat, g_feat], dim=1)).squeeze(1)


# ══════════════════════════════════════════════════════════════
#  5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_mae = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for img, gender, label in tqdm(loader, leave=False):
            img, gender, label = img.to(DEVICE), gender.to(DEVICE), label.to(DEVICE)
            pred = model(img, gender)
            loss = criterion(pred, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(label)
            # Denormalize MAE so it's reported in original months
            total_mae  += (torch.abs(pred - label) * AGE_SCALE).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_mae / n


def train_model(model_name: str, train_df: pd.DataFrame,
                val_df: pd.DataFrame) -> dict:
    cfg      = MODEL_REGISTRY[model_name]
    img_size = cfg["img_size"]

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  (img_size={img_size})")
    print(f"{'='*60}")

    train_ds = BoneAgeDataset(train_df, IMG_DIR, img_size=img_size, augment=True)
    val_ds   = BoneAgeDataset(val_df,   IMG_DIR, img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = BoneAgeModel(model_name).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Huber delta is also normalized: 10 months / AGE_SCALE
    criterion = nn.HuberLoss(delta=10.0 / AGE_SCALE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val_mae  = float("inf")
    patience_cnt  = 0
    ckpt_path     = OUTPUT_DIR / f"best_{model_name}.pth"

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_mae = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_mae = run_epoch(model, val_loader,   criterion)
        scheduler.step(vl_mae)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)

        print(f"  Ep {epoch:03d}/{NUM_EPOCHS} | "
              f"Train MAE: {tr_mae:.2f}m | Val MAE: {vl_mae:.2f}m")

        if vl_mae < best_val_mae:
            best_val_mae = vl_mae
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state": model.state_dict(),
                "best_val_mae": best_val_mae,
                "age_scale": AGE_SCALE,
            }, ckpt_path)
            print(f"    ✓ Checkpoint saved (val MAE: {best_val_mae:.2f} months)")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR / f"history_{model_name}.csv", index=False)

    print(f"\n  [{model_name}] Best Val MAE: {best_val_mae:.2f} months")
    return {"model_name": model_name, "best_val_mae": best_val_mae,
            "history": history, "ckpt_path": str(ckpt_path)}


# ══════════════════════════════════════════════════════════════
#  6. MAIN — Train All Models
# ══════════════════════════════════════════════════════════════

def main():
    global AGE_SCALE

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train (empty = all)")
    cli = parser.parse_args()

    # Override MODELS_TO_TRAIN if --models flag provided
    models_to_run = cli.models if cli.models else MODELS_TO_TRAIN
    unknown = [m for m in models_to_run if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. "
                         f"Valid options: {list(MODEL_REGISTRY.keys())}")

    df = pd.read_csv(CSV_PATH)
    df["male"] = df["male"].astype(bool)
    df["age_bucket"] = pd.cut(df["boneage"], bins=10, labels=False)

    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT,
        stratify=df["age_bucket"], random_state=SEED)
    print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")

    # Age normalization scale — derived from training split (same as top Kaggle notebooks)
    AGE_SCALE = 2.0 * float(train_df["boneage"].std())
    print(f"AGE_SCALE (2×std): {AGE_SCALE:.2f} months")
    pd.Series({"age_scale": AGE_SCALE}).to_json(OUTPUT_DIR / "age_scale.json")

    results = []
    for model_name in models_to_run:
        result = train_model(model_name, train_df, val_df)
        results.append({"Model": result["model_name"],
                        "Best_Val_MAE": result["best_val_mae"]})

    # Summary table
    summary_df = pd.DataFrame(results).sort_values("Best_Val_MAE")
    summary_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)

    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    print(f"\nBest model: {summary_df.iloc[0]['Model']} "
          f"(MAE: {summary_df.iloc[0]['Best_Val_MAE']:.2f} months)")


if __name__ == "__main__":
    main()
