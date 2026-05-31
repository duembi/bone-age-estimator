"""
Bone Age Predictor — 3 katmanlı iyileştirme:
  1. Ensemble  : 7 modelin ortalaması
  2. TTA       : Her görüntü 8 augmented versiyonda tahmin, ortalama alınır
  3. Calibration: Yaş grubuna göre sistematik bias düzeltmesi
"""

import io
import json
import base64
import numpy as np
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

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent          # Final Project/
RESULTS_DIR = BASE_DIR / "results"
CALIB_FILE  = Path(__file__).parent / "calibration.json"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REGISTRY = {
    "VGG16":          {"img_size": 256, "feature_dim": 512},
    "ResNet50":       {"img_size": 256, "feature_dim": 2048},
    "InceptionV3":    {"img_size": 299, "feature_dim": 2048},
    "DenseNet121":    {"img_size": 256, "feature_dim": 1024},
    "EfficientNetB0": {"img_size": 256, "feature_dim": 1280},
    "EfficientNetB3": {"img_size": 300, "feature_dim": 1536},
    "Xception":       {"img_size": 299, "feature_dim": 2048},
}

CLOSURE_AGE = {"male": 216, "female": 192}
MAX_AGE     = {"male": 228, "female": 216}

GROWTH_STAGES = [
    (0,   36,  "Erken Bebeklik",    "Kemikler hızla büyümektedir. Epifiz plakaları çok aktif."),
    (36,  60,  "Erken Çocukluk",    "Düzenli ve istikrarlı kemik büyümesi devam etmektedir."),
    (60,  120, "Okul Çağı",         "Büyüme hızı yavaşlamış ancak düzenli olarak devam etmektedir."),
    (120, 144, "Erken Ergenlik",    "Pubertenin başlangıcı, büyüme ivmelenmektedir."),
    (144, 180, "Orta Ergenlik",     "Zirve büyüme dönemi. Epifiz plakaları kapanmaya yaklaşıyor."),
    (180, 204, "Geç Ergenlik",      "Büyüme yavaşlıyor. Büyük büyüme plakaları kapanmaya başlıyor."),
    (204, 228, "Genç Yetişkin",     "İskelet gelişimi büyük ölçüde tamamlanmış."),
    (228, 999, "Yetişkin",          "İskelet gelişimi tamamen tamamlanmış."),
]


# ══════════════════════════════════════════════════════════════════════════════
#  1. MODEL TANIMLARI
# ══════════════════════════════════════════════════════════════════════════════

class _InceptionV3Features(nn.Module):
    def __init__(self, b): super().__init__(); self.b = b
    def forward(self, x):
        x = self.b.Conv2d_1a_3x3(x); x = self.b.Conv2d_2a_3x3(x)
        x = self.b.Conv2d_2b_3x3(x); x = self.b.maxpool1(x)
        x = self.b.Conv2d_3b_1x1(x); x = self.b.Conv2d_4a_3x3(x)
        x = self.b.maxpool2(x)
        x = self.b.Mixed_5b(x); x = self.b.Mixed_5c(x); x = self.b.Mixed_5d(x)
        x = self.b.Mixed_6a(x); x = self.b.Mixed_6b(x); x = self.b.Mixed_6c(x)
        x = self.b.Mixed_6d(x); x = self.b.Mixed_6e(x)
        x = self.b.Mixed_7a(x); x = self.b.Mixed_7b(x); x = self.b.Mixed_7c(x)
        return x

class _DenseNet121Features(nn.Module):
    def __init__(self, features): super().__init__(); self.features = features
    def forward(self, x): return F.relu(self.features(x), inplace=True)


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
    raise ValueError(f"Bilinmeyen model: {name}")


class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name, dropout=0.4):
        super().__init__()
        self.backbone_name = backbone_name
        image_branch, feature_dim = _build_backbone(backbone_name)
        self.image_branch  = image_branch
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


# ══════════════════════════════════════════════════════════════════════════════
#  2. MODEL CACHE (singleton per model name)
# ══════════════════════════════════════════════════════════════════════════════

_model_cache: dict = {}   # {name: (model, age_scale)}


def _load_single_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    ckpt_path = RESULTS_DIR / f"best_{name}.pth"
    if not ckpt_path.exists():
        return None
    try:
        ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        age_scale = ckpt.get("age_scale", 1.0)
        model     = BoneAgeModel(name).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[name] = (model, age_scale)
        print(f"  [Ensemble] {name} yuklendi (scale={age_scale:.2f})")
        return _model_cache[name]
    except Exception as e:
        print(f"  [Ensemble] {name} yuklenemedi: {e}")
        return None


def load_all_models() -> list:
    """Mevcut tüm checkpoint dosyalarını yükle, liste döndür."""
    loaded = []
    for name in MODEL_REGISTRY:
        result = _load_single_model(name)
        if result is not None:
            loaded.append((name, result[0], result[1]))
    if not loaded:
        raise FileNotFoundError(
            f"Hiç model bulunamadı: {RESULTS_DIR}\n"
            "results/best_*.pth dosyalarının mevcut olduğundan emin olun."
        )
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
#  3. PREPROCESSING + TTA
# ══════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _to_gray_square(image_bytes: bytes) -> np.ndarray:
    """Bytes → CLAHE uygulanmış kare gri görüntü (numpy uint8)."""
    arr  = np.frombuffer(image_bytes, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Gecersiz goruntu dosyasi. PNG veya JPEG yukleyin.")
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img    = clahe.apply(img)
    h, w   = img.shape
    size   = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = img
    return canvas


def _tensor_from_array(arr: np.ndarray, img_size: int) -> torch.Tensor:
    img = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = np.stack([img, img, img], axis=-1)
    tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return tf(Image.fromarray(img)).unsqueeze(0)


def _tta_tensors(gray_sq: np.ndarray, img_size: int) -> list:
    """
    8 TTA varyantı:
      0: orijinal
      1: yatay flip
      2: +10 derece rotasyon
      3: -10 derece rotasyon
      4: parlaklik +20%
      5: parlaklik -20%
      6: merkez crop %90
      7: yatay flip + +10 derece rotasyon
    """
    h, w = gray_sq.shape
    cx, cy = w // 2, h // 2
    variants = []

    def rot(img, angle):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

    def crop_center(img, pct=0.90):
        ch, cw = int(h * pct), int(w * pct)
        y0 = (h - ch) // 2; x0 = (w - cw) // 2
        return img[y0:y0+ch, x0:x0+cw]

    def brightness(img, factor):
        out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out

    flip = cv2.flip(gray_sq, 1)

    variants.append(gray_sq)
    variants.append(flip)
    variants.append(rot(gray_sq, 10))
    variants.append(rot(gray_sq, -10))
    variants.append(brightness(gray_sq, 1.20))
    variants.append(brightness(gray_sq, 0.80))
    variants.append(crop_center(gray_sq, 0.90))
    variants.append(rot(flip, 10))

    return [_tensor_from_array(v, img_size) for v in variants]


# ══════════════════════════════════════════════════════════════════════════════
#  4. KALİBRASYON
# ══════════════════════════════════════════════════════════════════════════════

# Varsayılan kalibrasyon (evaluate.py çalıştırılmadan önce kullanılır)
# Her grup için: düzeltme = -bias  (tahmin - gercek > 0 ise çıkar)
_DEFAULT_CALIB = {
    "groups": [
        {"lo": 0,   "hi": 24,  "correction": 0.0},
        {"lo": 24,  "hi": 72,  "correction": 0.0},
        {"lo": 72,  "hi": 144, "correction": 0.0},
        {"lo": 144, "hi": 999, "correction": 0.0},
    ],
    "linear": {"slope": 1.0, "intercept": 0.0},
    "fitted": False,
}


def load_calibration() -> dict:
    if CALIB_FILE.exists():
        with open(CALIB_FILE, "r") as f:
            return json.load(f)
    return _DEFAULT_CALIB


def save_calibration(calib: dict):
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"  Kalibrasyon kaydedildi: {CALIB_FILE}")


def apply_calibration(pred_months: float, calib: dict) -> float:
    if not calib.get("fitted"):
        return pred_months
    # Önce lineer düzeltme
    s = calib["linear"]["slope"]
    b = calib["linear"]["intercept"]
    calibrated = s * pred_months + b
    # Sonra grup bazlı ince ayar
    for g in calib["groups"]:
        if g["lo"] <= calibrated < g["hi"]:
            calibrated += g["correction"]
            break
    return float(np.clip(calibrated, 0, MAX_AGE["male"]))


def fit_calibration(trues: list, preds: list) -> dict:
    """
    İki aşamalı kalibrasyon:
    1. Lineer regresyon: pred → true (global eğim + offset)
    2. Grup bazlı artık bias düzeltmesi
    """
    t = np.array(trues, dtype=float)
    p = np.array(preds, dtype=float)

    # Lineer fit: p'yi t'ye hizala
    A  = np.vstack([p, np.ones(len(p))]).T
    slope, intercept = np.linalg.lstsq(A, t, rcond=None)[0]
    p_lin = slope * p + intercept

    # Grup bazlı artık bias
    groups = []
    for lo, hi in [(0, 24), (24, 72), (72, 144), (144, 999)]:
        mask = (t >= lo) & (t < hi)
        if mask.sum() > 0:
            residual_bias = float(np.mean(p_lin[mask] - t[mask]))
            groups.append({"lo": lo, "hi": hi, "correction": round(-residual_bias, 2)})
        else:
            groups.append({"lo": lo, "hi": hi, "correction": 0.0})

    calib = {
        "groups": groups,
        "linear": {"slope": round(float(slope), 6), "intercept": round(float(intercept), 4)},
        "fitted": True,
    }
    return calib


# ══════════════════════════════════════════════════════════════════════════════
#  5. KLİNİK YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def classify_growth_stage(months: float):
    for lo, hi, name, desc in GROWTH_STAGES:
        if lo <= months < hi:
            return name, desc
    return "Yetişkin", "İskelet gelişimi tamamen tamamlanmış."


def remaining_growth(months: float, is_male: bool) -> dict:
    gender      = "male" if is_male else "female"
    closure_age = CLOSURE_AGE[gender]
    max_age     = MAX_AGE[gender]
    predicted   = float(np.clip(months, 0, max_age))
    remaining   = max(0.0, closure_age - predicted)
    pct_complete = min(100.0, (predicted / closure_age) * 100)
    stage_name, stage_desc = classify_growth_stage(predicted)
    return {
        "total_remaining_months": round(remaining, 1),
        "remaining_years":        int(remaining // 12),
        "remaining_months":       int(remaining % 12),
        "closure_age_months":     closure_age,
        "pct_complete":           round(pct_complete, 1),
        "stage_name":             stage_name,
        "stage_desc":             stage_desc,
    }


def months_to_str(months: float) -> str:
    y = int(months) // 12
    m = int(months) % 12
    if y > 0 and m > 0:
        return f"{y} yıl {m} ay"
    elif y > 0:
        return f"{y} yıl"
    return f"{m} ay"


# ══════════════════════════════════════════════════════════════════════════════
#  6. ENSEMBLE + TTA TAHMİN
# ══════════════════════════════════════════════════════════════════════════════

def _predict_single_model(model, age_scale, img_tensors: list,
                           gender_val: float) -> float:
    """Bir model için TTA tahminlerinin ortalamasını döndür."""
    gender_t = torch.tensor([[gender_val]], dtype=torch.float32, device=DEVICE)
    preds = []
    with torch.no_grad():
        for t in img_tensors:
            raw  = model(t.to(DEVICE), gender_t).item()
            pred = float(raw * age_scale)
            preds.append(pred)
    return float(np.mean(preds))


def ensemble_tta_predict(image_bytes: bytes, is_male: bool,
                          use_calibration: bool = True) -> dict:
    """
    Tam pipeline:
      image_bytes → TTA tensors → ensemble (7 model) → kalibrasyon → sonuç
    """
    gender_val = 1.0 if is_male else 0.0
    calib      = load_calibration()

    gray_sq = _to_gray_square(image_bytes)
    loaded_models = load_all_models()

    model_preds_raw  = {}
    model_preds_cal  = {}

    for name, model, age_scale in loaded_models:
        cfg      = MODEL_REGISTRY[name]
        tensors  = _tta_tensors(gray_sq, cfg["img_size"])
        raw_pred = _predict_single_model(model, age_scale, tensors, gender_val)
        raw_pred = float(np.clip(raw_pred, 0, MAX_AGE["male"]))
        cal_pred = apply_calibration(raw_pred, calib) if use_calibration else raw_pred
        model_preds_raw[name] = round(raw_pred, 1)
        model_preds_cal[name] = round(cal_pred, 1)

    raw_vals  = np.array(list(model_preds_raw.values()))
    cal_vals  = np.array(list(model_preds_cal.values()))

    ensemble_raw = float(np.mean(raw_vals))
    ensemble_cal = float(np.mean(cal_vals))
    ensemble_std = float(np.std(cal_vals))

    final_pred = ensemble_cal if use_calibration else ensemble_raw

    return {
        "ensemble_mean":     round(final_pred, 1),
        "ensemble_raw":      round(ensemble_raw, 1),
        "ensemble_std":      round(ensemble_std, 1),
        "confidence_low":    round(max(0, final_pred - 1.96 * ensemble_std), 1),
        "confidence_high":   round(final_pred + 1.96 * ensemble_std, 1),
        "individual_raw":    model_preds_raw,
        "individual_cal":    model_preds_cal,
        "calibration_used":  use_calibration and calib.get("fitted", False),
        "n_models":          len(loaded_models),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  7. WEB APP GİRİŞ NOKTASI
# ══════════════════════════════════════════════════════════════════════════════

def predict_bone_age(image_bytes: bytes, gender: str) -> dict:
    is_male  = gender.lower() == "male"
    pred     = ensemble_tta_predict(image_bytes, is_male)
    bone_age = pred["ensemble_mean"]
    growth   = remaining_growth(bone_age, is_male)

    report_b64 = _generate_report_b64(image_bytes, is_male, pred, growth)

    return {
        "bone_age_months":  bone_age,
        "bone_age_str":     months_to_str(bone_age),
        "gender":           "Erkek" if is_male else "Kız",
        "pct_complete":     growth["pct_complete"],
        "stage_name":       growth["stage_name"],
        "stage_desc":       growth["stage_desc"],
        "remaining_str":    "Tamamlandı" if growth["total_remaining_months"] == 0
                            else months_to_str(growth["total_remaining_months"]),
        "remaining_months": growth["total_remaining_months"],
        "closure_str":      months_to_str(growth["closure_age_months"]),
        "n_models":         pred["n_models"],
        "calibrated":       pred["calibration_used"],
        "ensemble_std":     pred["ensemble_std"],
        "report_image":     report_b64,
        "max_age":          MAX_AGE["male" if is_male else "female"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  8. RAPOR GÖRSELİ
# ══════════════════════════════════════════════════════════════════════════════

def _generate_report_b64(image_bytes: bytes, is_male: bool,
                          pred: dict, growth: dict) -> str:
    bone_age   = pred["ensemble_mean"]
    std        = pred["ensemble_std"]
    ci_lo      = pred["confidence_low"]
    ci_hi      = pred["confidence_high"]
    indiv      = pred["individual_cal"] if pred["calibration_used"] else pred["individual_raw"]
    gender_str = "Erkek" if is_male else "Kız"
    gender_col = "#4C72B0" if is_male else "#DD8452"
    max_age    = MAX_AGE["male" if is_male else "female"]
    pct        = growth["pct_complete"]
    rem        = growth["total_remaining_months"]
    col        = "#27AE60" if rem == 0 else ("#E67E22" if rem < 24 else "#2980B9")

    n_models = pred["n_models"]
    calib_tag = f"Kalibrasyon: Aktif" if pred["calibration_used"] else "Kalibrasyon: Pasif"

    fig = plt.figure(figsize=(16, 10), facecolor="#F8F9FA")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35,
                            left=0.05, right=0.97, top=0.90, bottom=0.06)

    fig.text(0.5, 0.95, "KEMİK YAŞI TAHMİN RAPORU",
             ha="center", fontsize=15, fontweight="bold", color="#2C3E50")
    fig.text(0.5, 0.925,
             f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
             f"Ensemble ({n_models} model) + TTA (8x)  |  {calib_tag}",
             ha="center", fontsize=8.5, color="#7F8C8D")

    # X-ray
    ax_img = fig.add_subplot(gs[:, 0])
    arr  = np.frombuffer(image_bytes, dtype=np.uint8)
    xray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if xray is not None:
        ax_img.imshow(xray, cmap="bone")
    ax_img.set_title("El Bilek Röntgeni", fontweight="bold", fontsize=10)
    ax_img.axis("off")

    # Ana sonuç kutusu
    ax_main = fig.add_subplot(gs[0, 1:3])
    ax_main.axis("off")
    ax_main.add_patch(FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=gender_col, facecolor="#EBF5FB",
        transform=ax_main.transAxes, clip_on=False))
    ax_main.text(0.5, 0.84, f"Kemik Yaşı: {months_to_str(bone_age)}",
                 ha="center", fontsize=15, fontweight="bold",
                 color="#1A5276", transform=ax_main.transAxes)
    ax_main.text(0.5, 0.62, f"({bone_age:.1f} ay)   |   Cinsiyet: {gender_str}",
                 ha="center", fontsize=11, color="#2E86C1",
                 transform=ax_main.transAxes)
    ax_main.text(0.5, 0.38,
                 f"95% Güven Aralığı: {months_to_str(ci_lo)} — {months_to_str(ci_hi)}",
                 ha="center", fontsize=9, color="#808B96",
                 transform=ax_main.transAxes)
    ax_main.text(0.5, 0.18,
                 f"Model std: ±{std:.1f} ay   |   {n_models} model ensemble   |   TTA 8x",
                 ha="center", fontsize=8, color="#AAB7B8",
                 transform=ax_main.transAxes)

    # Kalan büyüme
    ax_rem = fig.add_subplot(gs[0, 3])
    ax_rem.axis("off")
    ax_rem.add_patch(FancyBboxPatch(
        (0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=col, facecolor="#EAFAF1",
        transform=ax_rem.transAxes, clip_on=False))
    ax_rem.text(0.5, 0.82, "Kalan Büyüme",
                ha="center", fontsize=11, fontweight="bold",
                color="#1E8449", transform=ax_rem.transAxes)
    rem_label = "Tamamlandı" if rem == 0 else months_to_str(rem)
    ax_rem.text(0.5, 0.57, rem_label,
                ha="center", fontsize=14, fontweight="bold",
                color=col, transform=ax_rem.transAxes)
    ax_rem.text(0.5, 0.30, f"İskelet Olgunluğu: %{pct:.1f}",
                ha="center", fontsize=9, color="#566573",
                transform=ax_rem.transAxes)

    # İlerleme çubuğu
    ax_prog = fig.add_subplot(gs[1, 1:])
    ax_prog.set_xlim(0, max_age)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_title("İskelet Olgunluk İlerleme Çubuğu", fontweight="bold", fontsize=10)
    ax_prog.barh(0.5, max_age, height=0.35, color="#D5DBDB", align="center", left=0)
    prog_col = plt.cm.RdYlGn(min(bone_age / max_age, 1.0))
    ax_prog.barh(0.5, bone_age, height=0.35, color=prog_col, align="center", left=0,
                 label=f"Tahmin: {bone_age:.1f} ay")
    ax_prog.barh(0.5, max(0, ci_hi - ci_lo), height=0.15, color="gray", alpha=0.5,
                 align="center", left=ci_lo,
                 label=f"95% GA: [{ci_lo:.0f}–{ci_hi:.0f}]")
    ax_prog.axvline(bone_age, color="#C0392B", lw=2, zorder=5)
    ax_prog.set_xticks([0, 24, 60, 120, 144, 180, 216, 228])
    ax_prog.set_xticklabels(["0", "2y", "5y", "10y", "12y", "15y", "18y", "19y"])
    ax_prog.set_yticks([])
    ax_prog.legend(loc="lower right", fontsize=8)

    # Büyüme evresi
    ax_stage = fig.add_subplot(gs[2, 1:3])
    ax_stage.axis("off")
    ax_stage.text(0.5, 0.80, f"Büyüme Evresi: {growth['stage_name']}",
                  ha="center", fontsize=11, fontweight="bold", color="#1A5276",
                  transform=ax_stage.transAxes)
    ax_stage.text(0.5, 0.48, growth["stage_desc"],
                  ha="center", fontsize=9.5, color="#566573",
                  transform=ax_stage.transAxes)
    ax_stage.text(0.5, 0.18,
                  f"Tahmini kapanma yaşı: ~{months_to_str(growth['closure_age_months'])}",
                  ha="center", fontsize=8.5, color="#808B96",
                  transform=ax_stage.transAxes)

    # Model tahminleri bar chart
    ax_models = fig.add_subplot(gs[2, 3])
    if indiv:
        names  = list(indiv.keys())
        vals   = list(indiv.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        bars   = ax_models.barh(names, vals, color=colors, edgecolor="white")
        ax_models.axvline(bone_age, color="red", lw=1.5, ls="--",
                          label=f"Ensemble: {bone_age:.1f}m")
        ax_models.set_title("Model Tahminleri", fontweight="bold", fontsize=9)
        ax_models.set_xlabel("Kemik Yaşı (ay)", fontsize=8)
        ax_models.tick_params(axis="y", labelsize=7)
        for bar, val in zip(bars, vals):
            ax_models.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                           f"{val:.1f}", va="center", fontsize=7)
        ax_models.legend(fontsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
