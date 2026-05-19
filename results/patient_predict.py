"""
RSNA Bone Age — Patient Inference System
Usage: python patient_predict.py --image path/to/xray.png --gender male
       python patient_predict.py --image path/to/xray.png --gender female
       python patient_predict.py --image path/to/xray.png --gender male --model EfficientNetB0

Output:
  - Bone age (months + years/months format)
  - Remaining bone development
  - Growth stage
  - Patient report image (patient_report.png)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import (
    VGG16_Weights, ResNet50_Weights, Inception_V3_Weights,
    DenseNet121_Weights, EfficientNet_B0_Weights, EfficientNet_B3_Weights,
)
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────── Paths ─────────────────────────────
OUTPUT_DIR = Path("/kaggle/working")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#  1. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

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
        try:
            import timm
            m = timm.create_model("xception", pretrained=False,
                                  num_classes=0, global_pool="")
            return m, 2048
        except ImportError:
            raise ImportError("timm is required for Xception: pip install timm")
    else:
        raise ValueError(f"Unknown model: {name}")


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
#  2. PREPROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess_image(img_path: str, img_size: int) -> torch.Tensor:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # Pad to square
    h, w  = img.shape
    size  = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = img

    # Resize & to 3-channel
    img   = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img   = np.stack([img, img, img], axis=-1)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(Image.fromarray(img)).unsqueeze(0)   # (1, 3, H, W)


# ══════════════════════════════════════════════════════════════
#  3. CLINICAL KNOWLEDGE — Remaining Growth
# ══════════════════════════════════════════════════════════════

# Growth plate closure references (months):
# Female: most plates fuse by ~192m (16y), complete ~216m (18y)
# Male:   most plates fuse by ~216m (18y), complete ~228m (19y)
CLOSURE_AGE = {"male": 216, "female": 192}
MAX_AGE     = {"male": 228, "female": 216}

# Tanner-aligned growth stages (in months)
GROWTH_STAGES = [
    (0,   36,  "Early Infancy",
     "Bones are growing rapidly. Epiphyseal plates are very active."),
    (36,  60,  "Early Childhood",
     "Regular and stable bone growth continues."),
    (60,  120, "School Age",
     "Growth rate has slowed but continues steadily."),
    (120, 144, "Early Adolescence (Tanner 2-3)",
     "Puberty onset, growth accelerating (beginning of growth spurt)."),
    (144, 180, "Mid Adolescence (Tanner 3-4)",
     "Peak growth period. Epiphyseal plates approaching closure."),
    (180, 204, "Late Adolescence (Tanner 4-5)",
     "Growth slowing. Major growth plates beginning to fuse."),
    (204, 228, "Late Adolescence / Young Adult",
     "Skeletal development largely complete."),
    (228, 999, "Adult",
     "Skeletal development fully complete."),
]


def classify_growth_stage(bone_age_months: float) -> tuple:
    """Return (stage_name, description, months_remaining, years, months_rem)."""
    for lo, hi, name, desc in GROWTH_STAGES:
        if lo <= bone_age_months < hi:
            return name, desc
    return "Adult", "Skeletal development fully complete."


def remaining_growth(bone_age_months: float, is_male: bool) -> dict:
    """
    Estimate remaining bone development based on predicted bone age and sex.

    Returns:
        total_remaining_months : total months of growth remaining
        remaining_years        : years component
        remaining_months       : months component
        closure_age_months     : expected age at growth plate closure
        pct_complete           : percentage of skeletal maturity achieved
        stage_name             : growth stage label
        stage_desc             : clinical description
    """
    gender       = "male" if is_male else "female"
    closure_age  = CLOSURE_AGE[gender]
    max_age      = MAX_AGE[gender]
    predicted    = float(np.clip(bone_age_months, 0, max_age))

    remaining    = max(0.0, closure_age - predicted)
    pct_complete = min(100.0, (predicted / closure_age) * 100)

    rem_years    = int(remaining // 12)
    rem_months   = int(remaining %  12)

    stage_name, stage_desc = classify_growth_stage(predicted)

    return {
        "total_remaining_months": round(remaining, 1),
        "remaining_years":        rem_years,
        "remaining_months":       rem_months,
        "closure_age_months":     closure_age,
        "pct_complete":           round(pct_complete, 1),
        "stage_name":             stage_name,
        "stage_desc":             stage_desc,
    }


def months_to_years_str(months: float) -> str:
    y = int(months) // 12
    m = int(months) % 12
    if y > 0 and m > 0:
        return f"{y} yr {m} mo"
    elif y > 0:
        return f"{y} yr"
    else:
        return f"{m} mo"


# ══════════════════════════════════════════════════════════════
#  4. INFERENCE
# ══════════════════════════════════════════════════════════════

def load_model(model_name: str, ckpt_path: Path) -> tuple:
    """Load model + age_scale from checkpoint. Returns (model, age_scale)."""
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    age_scale = ckpt.get("age_scale", 1.0)
    model     = BoneAgeModel(model_name).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, age_scale


def predict(model: BoneAgeModel, img_tensor: torch.Tensor,
            gender_val: float, age_scale: float) -> float:
    """Run inference and return predicted bone age in months."""
    gender_t = torch.tensor([[gender_val]], dtype=torch.float32, device=DEVICE)
    img_t    = img_tensor.to(DEVICE)
    with torch.no_grad():
        raw_pred = model(img_t, gender_t).item()
    return float(raw_pred * age_scale)


def ensemble_predict(image_path: str, is_male: bool) -> dict:
    """
    Run inference with all available trained models and return ensemble result.
    Returns individual predictions + ensemble mean + std.
    """
    gender_val  = 1.0 if is_male else 0.0
    model_preds = {}

    for model_name, cfg in MODEL_REGISTRY.items():
        ckpt_path = OUTPUT_DIR / f"best_{model_name}.pth"
        if not ckpt_path.exists():
            continue
        try:
            img_t        = preprocess_image(image_path, cfg["img_size"])
            model, scale = load_model(model_name, ckpt_path)
            pred_months  = predict(model, img_t, gender_val, scale)
            pred_months  = max(0.0, min(pred_months, MAX_AGE["male"]))
            model_preds[model_name] = round(pred_months, 1)
            del model   # free GPU memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"  [{model_name}] Error: {e}")

    if not model_preds:
        raise RuntimeError("No trained model checkpoints found in " + str(OUTPUT_DIR))

    vals     = np.array(list(model_preds.values()))
    ensemble = float(np.mean(vals))
    std      = float(np.std(vals))

    return {
        "individual":      model_preds,
        "ensemble_mean":   round(ensemble, 1),
        "ensemble_std":    round(std, 1),
        "confidence_low":  round(ensemble - 1.96 * std, 1),
        "confidence_high": round(ensemble + 1.96 * std, 1),
    }


# ══════════════════════════════════════════════════════════════
#  5. PATIENT REPORT VISUALIZATION
# ══════════════════════════════════════════════════════════════

def generate_patient_report(image_path: str,
                             is_male: bool,
                             pred_result: dict,
                             growth_info: dict,
                             output_path: Path) -> None:
    """Generate a comprehensive single-page patient report."""
    bone_age  = pred_result["ensemble_mean"]
    std       = pred_result["ensemble_std"]
    ci_lo     = pred_result["confidence_low"]
    ci_hi     = pred_result["confidence_high"]
    indiv     = pred_result["individual"]

    gender_str = "Male" if is_male else "Female"
    gender_col = "#4C72B0" if is_male else "#DD8452"

    fig = plt.figure(figsize=(18, 13), facecolor="#F8F9FA")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.05, right=0.97, top=0.92, bottom=0.06)

    # ── Header ──────────────────────────────────────────────────
    fig.text(0.5, 0.96, "BONE AGE PREDICTION REPORT",
             ha="center", fontsize=18, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.935, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ha="center", fontsize=10, color="#7F8C8D")

    # ── X-ray image ─────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[:, 0])
    try:
        xray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        ax_img.imshow(xray, cmap="bone")
        ax_img.set_title("Input Radiograph", fontweight="bold")
        ax_img.axis("off")
    except Exception:
        ax_img.text(0.5, 0.5, "Image\nnot loaded",
                    ha="center", va="center", transform=ax_img.transAxes)
        ax_img.axis("off")

    # ── Main result box ──────────────────────────────────────────
    ax_main = fig.add_subplot(gs[0, 1:3])
    ax_main.axis("off")
    ax_main.add_patch(FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.9,
        boxstyle="round,pad=0.02", linewidth=2,
        edgecolor=gender_col, facecolor="#EBF5FB",
        transform=ax_main.transAxes, clip_on=False))
    ax_main.text(0.5, 0.82, f"Bone Age: {months_to_years_str(bone_age)}",
                 ha="center", va="center", fontsize=16, fontweight="bold",
                 color="#1A5276", transform=ax_main.transAxes)
    ax_main.text(0.5, 0.62, f"({bone_age:.1f} months)",
                 ha="center", va="center", fontsize=13, color="#2E86C1",
                 transform=ax_main.transAxes)
    ax_main.text(0.5, 0.42, f"Gender: {gender_str}",
                 ha="center", va="center", fontsize=11, color="#566573",
                 transform=ax_main.transAxes)
    ax_main.text(0.5, 0.22, f"95% Confidence Interval: {months_to_years_str(ci_lo)} — {months_to_years_str(ci_hi)}",
                 ha="center", va="center", fontsize=9, color="#808B96",
                 transform=ax_main.transAxes)

    # ── Remaining growth box ─────────────────────────────────────
    ax_rem = fig.add_subplot(gs[0, 3])
    ax_rem.axis("off")
    rem   = growth_info["total_remaining_months"]
    pct   = growth_info["pct_complete"]
    color = "#27AE60" if rem == 0 else ("#E67E22" if rem < 24 else "#2980B9")
    ax_rem.add_patch(FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.9,
        boxstyle="round,pad=0.02", linewidth=2,
        edgecolor=color, facecolor="#EAFAF1",
        transform=ax_rem.transAxes, clip_on=False))
    ax_rem.text(0.5, 0.80, "Remaining Growth",
                ha="center", fontsize=11, fontweight="bold",
                color="#1E8449", transform=ax_rem.transAxes)
    if rem == 0:
        rem_str = "Complete"
    else:
        rem_str = months_to_years_str(rem)
    ax_rem.text(0.5, 0.55, rem_str,
                ha="center", fontsize=14, fontweight="bold",
                color=color, transform=ax_rem.transAxes)
    ax_rem.text(0.5, 0.30, f"Skeletal Maturity: {pct:.1f}%",
                ha="center", fontsize=9, color="#566573",
                transform=ax_rem.transAxes)

    # ── Skeletal maturity progress bar ───────────────────────────
    ax_prog = fig.add_subplot(gs[1, 1:])
    ax_prog.set_xlim(0, MAX_AGE["male"])
    ax_prog.set_ylim(0, 1)
    ax_prog.set_title("Skeletal Maturity Progress Bar", fontweight="bold")

    max_age = MAX_AGE["male" if is_male else "female"]
    # Background
    ax_prog.barh(0.5, max_age, height=0.35, color="#D5DBDB",
                 align="center", left=0)
    # Progress
    cmap_val = min(bone_age / max_age, 1.0)
    prog_color = plt.cm.RdYlGn(cmap_val)
    ax_prog.barh(0.5, bone_age, height=0.35, color=prog_color,
                 align="center", left=0, label=f"Predicted: {bone_age:.1f}m")
    # CI band
    ax_prog.barh(0.5, max(0, ci_hi - ci_lo), height=0.15,
                 color="gray", alpha=0.5, align="center", left=ci_lo,
                 label=f"95% CI: [{ci_lo:.0f}–{ci_hi:.0f}]")
    # Marker
    ax_prog.axvline(bone_age, color="#C0392B", lw=2, zorder=5)
    # Age ticks
    ticks = [0, 24, 60, 120, 144, 180, 216, 228]
    labels = ["0", "2y", "5y", "10y", "12y", "15y", "18y", "19y"]
    ax_prog.set_xticks(ticks)
    ax_prog.set_xticklabels(labels)
    ax_prog.set_yticks([])
    ax_prog.legend(loc="lower right", fontsize=8)

    # ── Growth stage info ────────────────────────────────────────
    ax_stage = fig.add_subplot(gs[2, 1:3])
    ax_stage.axis("off")
    ax_stage.text(0.5, 0.85, f"Growth Stage: {growth_info['stage_name']}",
                  ha="center", fontsize=12, fontweight="bold", color="#1A5276",
                  transform=ax_stage.transAxes)
    ax_stage.text(0.5, 0.55, growth_info["stage_desc"],
                  ha="center", fontsize=10, color="#566573",
                  transform=ax_stage.transAxes, wrap=True)
    closure_str = months_to_years_str(growth_info["closure_age_months"])
    ax_stage.text(0.5, 0.25,
                  f"Estimated growth plate closure: ~{closure_str}  |  "
                  f"Model ensemble std: ±{std:.1f} months",
                  ha="center", fontsize=9, color="#808B96",
                  transform=ax_stage.transAxes)

    # ── Individual model predictions bar chart ───────────────────
    ax_models = fig.add_subplot(gs[2, 3])
    if len(indiv) > 0:
        model_names = list(indiv.keys())
        model_vals  = list(indiv.values())
        colors      = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        bars = ax_models.barh(model_names, model_vals, color=colors, edgecolor="white")
        ax_models.axvline(bone_age, color="red", lw=1.5, ls="--",
                          label=f"Ensemble: {bone_age:.1f}m")
        ax_models.set_title("Model Predictions", fontweight="bold", fontsize=9)
        ax_models.set_xlabel("Bone Age (months)", fontsize=8)
        ax_models.tick_params(axis="y", labelsize=7)
        for bar, val in zip(bars, model_vals):
            ax_models.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                           f"{val:.1f}", va="center", fontsize=7)
        ax_models.legend(fontsize=7)

    # ── Save ─────────────────────────────────────────────────────
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Report saved → {output_path}")


# ══════════════════════════════════════════════════════════════
#  6. MAIN CLI
# ══════════════════════════════════════════════════════════════

def main(image=None, gender=None, model="ensemble", output=None):
    # ── CLI mode uses argparse; notebook mode uses direct parameters ──
    if image is None:
        if len(sys.argv) > 1:
            parser = argparse.ArgumentParser(
                description="Bone Age Prediction System — RSNA Bone Age")
            parser.add_argument("--image",  required=True,
                                help="X-ray image path (PNG/JPG)")
            parser.add_argument("--gender", required=True, choices=["male", "female"],
                                help="Gender: male / female")
            parser.add_argument("--model",  default="ensemble",
                                help="Model name or 'ensemble' (default)")
            parser.add_argument("--output", default=None,
                                help="Report output path (default: patient_report.png)")
            args   = parser.parse_args()
            image  = args.image
            gender = args.gender
            model  = args.model
            output = args.output
        else:
            raise ValueError("image and gender parameters are required. Example: main(image='path.png', gender='male')")

    # ── Validate inputs ──────────────────────────────────────────
    image_path = Path(image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    is_male  = gender.lower() in ("male",)
    output_p = Path(output) if output else OUTPUT_DIR / "patient_report.png"

    print("=" * 60)
    print("  BONE AGE PREDICTION SYSTEM")
    print("=" * 60)
    print(f"  Image  : {image_path}")
    print(f"  Gender : {'Male' if is_male else 'Female'}")
    print(f"  Model  : {model}")
    print()

    # ── Run prediction ───────────────────────────────────────────
    if model == "ensemble":
        print("Running ensemble prediction (all available models)...")
        pred_result = ensemble_predict(str(image_path), is_male)
    else:
        if model not in MODEL_REGISTRY:
            print(f"ERROR: Unknown model '{model}'. "
                  f"Options: {list(MODEL_REGISTRY.keys())} or 'ensemble'")
            sys.exit(1)
        ckpt_path = OUTPUT_DIR / f"best_{model}.pth"
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        print(f"Loading model {model}...")
        cfg          = MODEL_REGISTRY[model]
        img_t        = preprocess_image(str(image_path), cfg["img_size"])
        mdl, scale   = load_model(model, ckpt_path)
        pred_months  = predict(mdl, img_t, 1.0 if is_male else 0.0, scale)
        pred_months  = max(0.0, min(pred_months, MAX_AGE["male"]))
        pred_result  = {
            "individual":      {model: round(pred_months, 1)},
            "ensemble_mean":   round(pred_months, 1),
            "ensemble_std":    0.0,
            "confidence_low":  round(pred_months, 1),
            "confidence_high": round(pred_months, 1),
        }

    # ── Remaining growth ─────────────────────────────────────────
    bone_age    = pred_result["ensemble_mean"]
    growth_info = remaining_growth(bone_age, is_male)

    # ── Print results ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Bone Age          : {months_to_years_str(bone_age)} ({bone_age} months)")
    print(f"  95% Conf. Interval: {months_to_years_str(pred_result['confidence_low'])} — "
          f"{months_to_years_str(pred_result['confidence_high'])}")
    print()
    print(f"  Growth Stage      : {growth_info['stage_name']}")
    print(f"  Description       : {growth_info['stage_desc']}")
    print()
    print(f"  Skeletal Maturity : {growth_info['pct_complete']:.1f}% complete")
    if growth_info["total_remaining_months"] > 0:
        print(f"  Remaining Growth  : {months_to_years_str(growth_info['total_remaining_months'])}")
        print(f"  Est. Closure      : ~{months_to_years_str(growth_info['closure_age_months'])}")
    else:
        print(f"  Remaining Growth  : Skeletal development complete")
    print()

    if len(pred_result["individual"]) > 1:
        print("  Model Predictions:")
        for model_name, val in sorted(pred_result["individual"].items()):
            print(f"    {model_name:15s}: {val:.1f} months ({months_to_years_str(val)})")
        print(f"  Ensemble Mean     : {bone_age:.1f} months  (std: {pred_result['ensemble_std']:.1f})")

    # ── Generate report ──────────────────────────────────────────
    print()
    print("Generating patient report image...")
    generate_patient_report(
        str(image_path), is_male, pred_result, growth_info, output_p)

    print()
    print("=" * 60)
    print(f"  Report saved: {output_p}")
    print("=" * 60)

    return pred_result, growth_info, output_p


IMAGE_PATH = "/kaggle/input/datasets/kmader/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/9273.png"

main(image=IMAGE_PATH, gender="male")

import IPython.display as disp
disp.display(disp.Image("/kaggle/working/patient_report.png"))
