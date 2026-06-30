"""
Phase 3 - Step 3: Evaluasi Model YOLO11
========================================
Script ini mengevaluasi model hasil training pada validation set
dan menampilkan metrik: mAP, Precision, Recall, F1 Score.

Jalankan dari root project:
    python training/3_evaluate.py

Atau tentukan model secara manual:
    python training/3_evaluate.py --model path/ke/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ── Konfigurasi Default ───────────────────────────────────────────────────────

RUN_NAME = "receipt_yolo11"
DATA_YAML = Path("dataset/data.yaml")
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.6


# ── Helpers ───────────────────────────────────────────────────────────────────


def find_best_model(run_name: str) -> Path | None:
    """Cari best.pt secara dinamis di seluruh subfolder runs/."""
    if not Path("runs").exists():
        return None
    # Cari berdasarkan nama run
    candidates = list(Path("runs").rglob(f"{run_name}/weights/best.pt"))
    if candidates:
        return candidates[0]
    # Fallback: ambil best.pt manapun yang ada
    candidates = list(Path("runs").rglob("best.pt"))
    if candidates:
        return candidates[0]
    return None


def parse_args():
    # Cari model dulu untuk default argumen
    found_model = find_best_model(RUN_NAME)
    default_model = (
        str(found_model)
        if found_model
        else "runs/detect/receipt_yolo11/weights/best.pt"
    )

    parser = argparse.ArgumentParser(description="Evaluasi Model YOLO11")
    parser.add_argument("--model", default=default_model, help="Path ke file model .pt")
    parser.add_argument("--data", default=str(DATA_YAML), help="Path ke data.yaml")
    parser.add_argument("--imgsz", default=IMG_SIZE, type=int)
    parser.add_argument("--conf", default=CONF_THRESH, type=float)
    parser.add_argument("--iou", default=IOU_THRESH, type=float)
    parser.add_argument("--device", default="", help="cuda device atau cpu")
    return parser.parse_args()


def print_metrics(metrics) -> None:
    """Tampilkan metrik evaluasi secara rapi."""
    names = metrics.names  # dict {0: 'item', 1: 'price', ...}

    print()
    print("=" * 55)
    print("  Hasil Evaluasi Model")
    print("=" * 55)

    results_dict = metrics.results_dict

    map50 = results_dict.get("metrics/mAP50(B)", 0)
    map5095 = results_dict.get("metrics/mAP50-95(B)", 0)
    prec = results_dict.get("metrics/precision(B)", 0)
    recall = results_dict.get("metrics/recall(B)", 0)
    f1 = 2 * (prec * recall) / (prec + recall + 1e-9)

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 30}")
    print(f"  {'mAP@0.50':<20} {map50:>10.4f}")
    print(f"  {'mAP@0.50:0.95':<20} {map5095:>10.4f}")
    print(f"  {'Precision':<20} {prec:>10.4f}")
    print(f"  {'Recall':<20} {recall:>10.4f}")
    print(f"  {'F1 Score':<20} {f1:>10.4f}")
    print()

    # Per-class breakdown
    if hasattr(metrics, "box") and hasattr(metrics.box, "ap_class_index"):
        print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'mAP50':>10}")
        print(f"  {'-' * 45}")
        ap50_per_class = metrics.box.ap50
        p_per_class = metrics.box.p
        r_per_class = metrics.box.r
        class_indices = metrics.box.ap_class_index

        for i, cls_idx in enumerate(class_indices):
            cls_name = names.get(int(cls_idx), str(cls_idx))
            p_val = p_per_class[i] if i < len(p_per_class) else 0.0
            r_val = r_per_class[i] if i < len(r_per_class) else 0.0
            ap_val = ap50_per_class[i] if i < len(ap50_per_class) else 0.0
            print(f"  {cls_name:<12} {p_val:>10.4f} {r_val:>10.4f} {ap_val:>10.4f}")

    print()
    print("  Hasil lengkap tersimpan di: runs/val/")
    print("=" * 55)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        print("        Jalankan dulu: python training/2_train.py")
        return

    if not data_path.exists():
        print(f"[ERROR] data.yaml tidak ditemukan: {data_path}")
        return

    print(f"[INFO] Memuat model: {model_path}")
    model = YOLO(str(model_path))

    print(f"[INFO] Mengevaluasi pada validation set...")
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        plots=True,
        verbose=False,
    )

    print_metrics(metrics)


if __name__ == "__main__":
    main()
