"""
Phase 3 - Step 2: YOLO11 Training
===================================
Script ini menjalankan training YOLO11 pada dataset receipt.

Jalankan dari root project:
    python training/2_train.py

Atau dengan override argumen:
    python training/2_train.py --epochs 100 --batch 8 --model yolo11s.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ── Konfigurasi Default ───────────────────────────────────────────────────────

DATA_YAML = Path("dataset/data.yaml")
MODEL = "yolo11n.pt"  # n=nano, s=small, m=medium, l=large, x=xlarge
EPOCHS = 100
BATCH = 8  # Sesuaikan dengan VRAM GPU Anda
IMG_SIZE = 640
PROJECT_DIR = Path("runs/detect")
RUN_NAME = "receipt_yolo11"

# Augmentasi khusus untuk dokumen/struk
AUG_CONFIG = dict(
    hsv_h=0.01,  # variasi warna minimal (struk biasanya hitam-putih)
    hsv_s=0.2,
    hsv_v=0.3,
    degrees=3.0,  # rotasi kecil (struk tidak terlalu miring)
    translate=0.05,
    scale=0.3,
    shear=2.0,
    flipud=0.0,  # struk tidak dibalik vertikal
    fliplr=0.0,  # struk tidak dibalik horizontal
    mosaic=0.5,
    mixup=0.0,
)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 Receipt Training")
    parser.add_argument(
        "--model", default=MODEL, help="Model YOLO11 (yolo11n/s/m/l/x.pt)"
    )
    parser.add_argument("--data", default=str(DATA_YAML), help="Path ke data.yaml")
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--batch", default=BATCH, type=int)
    parser.add_argument("--imgsz", default=IMG_SIZE, type=int)
    parser.add_argument("--device", default="", help="cuda device (0, 0,1, cpu)")
    parser.add_argument(
        "--resume", action="store_true", help="Lanjutkan training dari checkpoint"
    )
    parser.add_argument("--name", default=RUN_NAME, help="Nama run/experiment")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  Phase 3 - YOLO11 Receipt Training")
    print("=" * 55)
    print(f"  Model   : {args.model}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Img Size: {args.imgsz}")
    print(f"  Device  : {args.device if args.device else 'auto'}")
    print("=" * 55)

    # Validasi data.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] data.yaml tidak ditemukan: {data_path}")
        print("        Jalankan dulu: python training/1_prepare_dataset.py")
        return

    # Load model
    if args.resume:
        # Lanjutkan dari last checkpoint
        last_ckpt = Path(PROJECT_DIR) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"[ERROR] Checkpoint tidak ditemukan: {last_ckpt}")
            return
        print(f"[INFO] Melanjutkan training dari: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"[INFO] Memuat pretrained model: {args.model}")
        model = YOLO(args.model)

    # Mulai training
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        name=args.name,
        exist_ok=args.resume,
        # Optimasi
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=20,  # Early stopping jika tidak ada peningkatan
        save_period=10,  # Simpan checkpoint setiap N epoch
        # Augmentasi
        **AUG_CONFIG,
        # Logging
        plots=True,
        verbose=True,
    )

    print()
    print("[DONE] Training selesai!")
    # Cari best.pt secara dinamis karena YOLO menentukan path-nya sendiri
    best_models = list(Path("runs").rglob(f"{args.name}/weights/best.pt"))
    if best_models:
        print(f"       Best model: {best_models[0]}")
    print(f"       Untuk evaluasi: python training/3_evaluate.py")


if __name__ == "__main__":
    main()
