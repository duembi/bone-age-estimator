"""
3 asamali basarim degerlendirmesi:
  Asama 1: Sadece EfficientNetB0 (baseline)
  Asama 2: Ensemble + TTA (kalibrasyon yok)
  Asama 3: Ensemble + TTA + Kalibrasyon (tam pipeline)

Kullanim:
  python evaluate.py --data /path/to/test/images
  python evaluate.py --data /path/to/test/images --gender female

Goruntu dosya adi kurali  (Turkce):
  Xayerkek / Xaykiz          -> X ay
  Xyaserkek / Xyaskiz        -> X yil
  Xbucukyaserkek / ...kiz    -> X.5 yil
"""

import argparse
import sys
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent


# ── Argumanlar ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Kemik yasi sistemi degerlendirme")
    p.add_argument("--data",   required=True,
                   help="Test goruntu klasoru (dosya adlari yas bilgisi icermeli)")
    p.add_argument("--gender", default="male", choices=["male", "female"],
                   help="Goruntulerin cinsiyeti (varsayilan: male)")
    return p.parse_args()


# ── Dosya adindan gercek yasi cikar ──────────────────────────────────────────
def parse_age_months(filename: str):
    stem = filename.lower().replace(".jpg","").replace(".png","")
    stem = stem.replace("erkek","").replace("kiz","")
    m = re.match(r"^(\d+)bucukyas$", stem)
    if m: return int(m.group(1)) * 12 + 6
    m = re.match(r"^(\d+)yas$", stem)
    if m: return int(m.group(1)) * 12
    m = re.match(r"^(\d+)ay$", stem)
    if m: return float(m.group(1))
    return None


def months_to_str(months: float) -> str:
    y = int(months) // 12; m = int(months) % 12
    if y > 0 and m > 0: return f"{y}y {m}m"
    if y > 0: return f"{y}y"
    return f"{m}m"


def metrics(results: list) -> dict:
    errs = np.array([r["error"] for r in results])
    abs_errs = np.abs(errs)
    trues = np.array([r["true"] for r in results])
    preds = np.array([r["pred"] for r in results])
    corr = float(np.corrcoef(trues, preds)[0,1]) if len(results) > 1 else 0
    within_1y = np.mean(abs_errs <= 12) * 100
    within_2y = np.mean(abs_errs <= 24) * 100
    group_maes = {}
    for lo, hi, lbl in [(0,24,"0-2y"),(24,72,"2-6y"),(72,144,"6-12y"),(144,999,"12+y")]:
        sub = [r["abs_error"] for r in results if lo <= r["true"] < hi]
        group_maes[lbl] = round(float(np.mean(sub)), 1) if sub else None
    return {
        "mae":       round(float(np.mean(abs_errs)), 2),
        "rmse":      round(float(np.sqrt(np.mean(errs**2))), 2),
        "bias":      round(float(np.mean(errs)), 2),
        "corr":      round(corr, 3),
        "within_1y": round(within_1y, 1),
        "within_2y": round(within_2y, 1),
        "n":         len(results),
        "group_maes": group_maes,
    }


# ── Asama 1: Sadece EfficientNetB0 baseline ──────────────────────────────────
def run_baseline(images: list, is_male: bool) -> list:
    print("\n--- ASAMA 1: EfficientNetB0 baseline (TTA yok, kalibrasyon yok) ---")
    from predictor import (RESULTS_DIR, DEVICE, MODEL_REGISTRY,
                            BoneAgeModel, _to_gray_square, _tensor_from_array)
    import torch

    name = "EfficientNetB0"
    cfg  = MODEL_REGISTRY[name]
    ckpt_path = RESULTS_DIR / f"best_{name}.pth"
    ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    age_scale = ckpt.get("age_scale", 1.0)
    model     = BoneAgeModel(name).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gender_val = 1.0 if is_male else 0.0
    results = []
    for img_path in images:
        true_m = parse_age_months(img_path.name)
        if true_m is None: continue
        try:
            with open(img_path, "rb") as f: image_bytes = f.read()
            gray_sq  = _to_gray_square(image_bytes)
            t        = _tensor_from_array(gray_sq, cfg["img_size"]).to(DEVICE)
            gender_t = torch.tensor([[gender_val]], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                raw = model(t, gender_t).item()
            pred_m = float(np.clip(raw * age_scale, 0, 228))
            err    = pred_m - true_m
            results.append({"file": img_path.name, "true": true_m, "pred": pred_m,
                             "error": err, "abs_error": abs(err)})
        except Exception as e:
            print(f"  [HATA] {img_path.name}: {e}")
    return results


# ── Asama 2: Ensemble + TTA (kalibrasyon olmadan) ────────────────────────────
def run_ensemble_tta(images: list, is_male: bool) -> list:
    print("\n--- ASAMA 2: Ensemble (7 model) + TTA (8x) — kalibrasyon yok ---")
    from predictor import ensemble_tta_predict, _DEFAULT_CALIB, save_calibration
    save_calibration(_DEFAULT_CALIB)

    results = []
    for i, img_path in enumerate(images):
        true_m = parse_age_months(img_path.name)
        if true_m is None: continue
        try:
            with open(img_path, "rb") as f: image_bytes = f.read()
            pred   = ensemble_tta_predict(image_bytes, is_male=is_male, use_calibration=False)
            pred_m = pred["ensemble_mean"]
            err    = pred_m - true_m
            results.append({"file": img_path.name, "true": true_m, "pred": pred_m,
                             "error": err, "abs_error": abs(err),
                             "n_models": pred["n_models"]})
            print(f"  [{i+1:02d}/{len(images)}] {img_path.name:<25} "
                  f"gercek={months_to_str(true_m):>7} "
                  f"tahmin={months_to_str(pred_m):>7} "
                  f"hata={err:>+7.1f}m")
        except Exception as e:
            print(f"  [HATA] {img_path.name}: {e}")
    return results


# ── Asama 3: Kalibrasyon fit et + tam pipeline ────────────────────────────────
def run_with_calibration(images: list, is_male: bool, phase2_results: list) -> list:
    print("\n--- ASAMA 3: Kalibrasyon fit ediliyor ve uygulanıyor ---")
    from predictor import fit_calibration, save_calibration, ensemble_tta_predict

    trues = [r["true"] for r in phase2_results]
    preds = [r["pred"] for r in phase2_results]
    calib = fit_calibration(trues, preds)
    save_calibration(calib)
    print(f"  Lineer fit: slope={calib['linear']['slope']:.4f}, "
          f"intercept={calib['linear']['intercept']:.2f}")
    for g in calib["groups"]:
        print(f"  Grup {g['lo']}-{g['hi']} ay: duzeltme = {g['correction']:+.1f} ay")

    print("\n  Kalibrasyonlu tahminler calistirilıyor...")
    results = []
    for i, img_path in enumerate(images):
        true_m = parse_age_months(img_path.name)
        if true_m is None: continue
        try:
            with open(img_path, "rb") as f: image_bytes = f.read()
            pred   = ensemble_tta_predict(image_bytes, is_male=is_male, use_calibration=True)
            pred_m = pred["ensemble_mean"]
            err    = pred_m - true_m
            results.append({"file": img_path.name, "true": true_m, "pred": pred_m,
                             "error": err, "abs_error": abs(err)})
            print(f"  [{i+1:02d}/{len(images)}] {img_path.name:<25} "
                  f"gercek={months_to_str(true_m):>7} "
                  f"tahmin={months_to_str(pred_m):>7} "
                  f"hata={err:>+7.1f}m")
        except Exception as e:
            print(f"  [HATA] {img_path.name}: {e}")
    return results


# ── Karsilastirma grafigi ─────────────────────────────────────────────────────
def plot_comparison(r1, r2, r3, m1, m2, m3):
    fig = plt.figure(figsize=(20, 14), facecolor="#F8F9FA")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
                            left=0.06, right=0.97, top=0.91, bottom=0.07)
    fig.text(0.5, 0.95, "Kemik Yasi Sistemi - Uc Asama Karsilastirma Raporu",
             ha="center", fontsize=14, fontweight="bold", color="#2C3E50")

    labels = ["Asama 1\nEfficientNetB0", "Asama 2\nEnsemble+TTA", "Asama 3\nEnsemble+TTA+Kalib."]
    colors = ["#E74C3C", "#E67E22", "#27AE60"]
    metrics_list = [m1, m2, m3]
    results_list = [r1, r2, r3]

    for col_idx, (res, m, lbl, col) in enumerate(zip(results_list, metrics_list, labels, colors)):
        ax = fig.add_subplot(gs[0, col_idx])
        t  = np.array([r["true"] for r in res])
        p  = np.array([r["pred"] for r in res])
        ax.scatter(t, p, c=col, s=70, zorder=3, edgecolors="white", lw=0.5, alpha=0.85)
        mn, mx = min(t.min(), p.min()) - 5, max(t.max(), p.max()) + 5
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.4)
        ax.fill_between([mn,mx],[mn-12,mx-12],[mn+12,mx+12], alpha=0.1, color="green")
        ax.fill_between([mn,mx],[mn-24,mx-24],[mn+24,mx+24], alpha=0.07, color="orange")
        ax.set_xlabel("Gercek (ay)", fontsize=8); ax.set_ylabel("Tahmin (ay)", fontsize=8)
        ax.set_title(f"{lbl}\nMAE={m['mae']:.1f}m  r={m['corr']:.3f}",
                     fontweight="bold", fontsize=9, color=col)
        ax.grid(True, alpha=0.2)

    for col_idx, (res, m, lbl, col) in enumerate(zip(results_list, metrics_list, labels, colors)):
        ax = fig.add_subplot(gs[1, col_idx])
        errs = [r["error"] for r in res]
        ax.hist(errs, bins=10, color=col, edgecolor="white", alpha=0.8)
        ax.axvline(0, color="k", lw=1.5, ls="--")
        ax.axvline(m["bias"], color="#C0392B", lw=1.5, label=f"Bias={m['bias']:+.1f}m")
        ax.set_xlabel("Hata (ay)", fontsize=8); ax.set_ylabel("Frekans", fontsize=8)
        ax.set_title(f"Hata Dagilimi\nBias={m['bias']:+.1f}m  RMSE={m['rmse']:.1f}m",
                     fontweight="bold", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax_bar = fig.add_subplot(gs[2, 0])
    metric_names = ["MAE (ay)", "RMSE (ay)", "Bias (ay)"]
    x = np.arange(len(metric_names))
    w = 0.25
    for i, (m, lbl, col) in enumerate(zip(metrics_list, labels, colors)):
        vals = [m["mae"], m["rmse"], abs(m["bias"])]
        bars = ax_bar.bar(x + i*w, vals, w, label=lbl.replace("\n"," "),
                          color=col, edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f"{v:.1f}", ha="center", fontsize=7)
    ax_bar.set_xticks(x + w); ax_bar.set_xticklabels(metric_names, fontsize=8)
    ax_bar.set_ylabel("Ay"); ax_bar.set_title("Hata Metrikleri Karsilastirma", fontweight="bold")
    ax_bar.legend(fontsize=7); ax_bar.grid(True, alpha=0.2, axis="y")

    ax_acc = fig.add_subplot(gs[2, 1])
    acc_labels = [l.replace("\n"," ") for l in labels]
    w1y = [m["within_1y"] for m in metrics_list]
    w2y = [m["within_2y"] for m in metrics_list]
    x2  = np.arange(len(acc_labels))
    b1  = ax_acc.bar(x2 - 0.2, w1y, 0.35, label="+/-1 yil", color=colors, edgecolor="white", alpha=0.85)
    b2  = ax_acc.bar(x2 + 0.2, w2y, 0.35, label="+/-2 yil", color=colors, edgecolor="white", alpha=0.45)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{h:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax_acc.axhline(99.9, color="gold", lw=1.5, ls="--", label="Hedef %99.9")
    ax_acc.set_ylim(0, 115)
    ax_acc.set_xticks(x2); ax_acc.set_xticklabels(acc_labels, fontsize=7)
    ax_acc.set_ylabel("Yuzde (%)"); ax_acc.set_title("+/-1y ve +/-2y Dogruluk", fontweight="bold")
    ax_acc.legend(fontsize=7); ax_acc.grid(True, alpha=0.2, axis="y")

    ax_grp = fig.add_subplot(gs[2, 2])
    grp_keys = ["0-2y", "2-6y", "6-12y", "12+y"]
    x3 = np.arange(len(grp_keys))
    for i, (m, lbl, col) in enumerate(zip(metrics_list, labels, colors)):
        gm = [m["group_maes"].get(k) or 0 for k in grp_keys]
        ax_grp.plot(x3, gm, "o-", color=col, lw=2, ms=7,
                    label=lbl.replace("\n"," "), alpha=0.85)
    ax_grp.axhline(12, color="#27AE60", lw=1, ls="--", label="+/-1 yil siniri")
    ax_grp.set_xticks(x3); ax_grp.set_xticklabels(grp_keys)
    ax_grp.set_ylabel("MAE (ay)"); ax_grp.set_title("Yas Grubuna Gore MAE", fontweight="bold")
    ax_grp.legend(fontsize=7); ax_grp.grid(True, alpha=0.2)

    out = OUTPUT_DIR / "eval_comparison.png"
    fig.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Grafik kaydedildi: {out}")


# ── Ozet tablo ────────────────────────────────────────────────────────────────
def print_summary(m1, m2, m3):
    sep = "=" * 70
    print(f"\n{sep}\n  SONUC OZETI\n{sep}")
    fmt = "  {:<28} {:>10} {:>10} {:>10}"
    print(fmt.format("Metrik", "Baseline", "Ens+TTA", "Ens+TTA+Kal"))
    print("-" * 70)
    def imp(old, new, lower=True):
        d = new - old
        return f"({'-' if (d<=0)==lower else '+'}{abs(d):.1f})"
    print(fmt.format("MAE (ay)", f"{m1['mae']:.1f}",
                     f"{m2['mae']:.1f} {imp(m1['mae'],m2['mae'])}",
                     f"{m3['mae']:.1f} {imp(m1['mae'],m3['mae'])}"))
    print(fmt.format("RMSE (ay)", f"{m1['rmse']:.1f}",
                     f"{m2['rmse']:.1f} {imp(m1['rmse'],m2['rmse'])}",
                     f"{m3['rmse']:.1f} {imp(m1['rmse'],m3['rmse'])}"))
    print(fmt.format("Bias (ay)", f"{m1['bias']:+.1f}", f"{m2['bias']:+.1f}", f"{m3['bias']:+.1f}"))
    print(fmt.format("Korelasyon (r)", f"{m1['corr']:.3f}", f"{m2['corr']:.3f}", f"{m3['corr']:.3f}"))
    print(fmt.format("+/-1 yil", f"%{m1['within_1y']:.1f}",
                     f"%{m2['within_1y']:.1f} {imp(m1['within_1y'],m2['within_1y'],False)}",
                     f"%{m3['within_1y']:.1f} {imp(m1['within_1y'],m3['within_1y'],False)}"))
    print(fmt.format("+/-2 yil", f"%{m1['within_2y']:.1f}",
                     f"%{m2['within_2y']:.1f}", f"%{m3['within_2y']:.1f}"))
    print(sep)
    print(f"\n  MAE iyilesmesi : {m1['mae']:.1f} -> {m3['mae']:.1f} ay  "
          f"({(1-m3['mae']/m1['mae'])*100:.1f}% azalma)")
    print(f"  +/-1y dogruluk : %{m1['within_1y']:.1f} -> %{m3['within_1y']:.1f}")
    print(sep)


# ── Ana ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args    = parse_args()
    data_dir = Path(args.data)
    is_male  = args.gender == "male"

    if not data_dir.exists():
        print(f"[HATA] Klasor bulunamadi: {data_dir}")
        sys.exit(1)

    images = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))
    images = [img for img in images if parse_age_months(img.name) is not None]
    print(f"Test goruntu sayisi: {len(images)}  |  Cinsiyet: {args.gender}")

    r1 = run_baseline(images, is_male);   m1 = metrics(r1)
    r2 = run_ensemble_tta(images, is_male); m2 = metrics(r2)
    r3 = run_with_calibration(images, is_male, r2); m3 = metrics(r3)

    print_summary(m1, m2, m3)
    plot_comparison(r1, r2, r3, m1, m2, m3)
