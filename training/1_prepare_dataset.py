"""
Phase 3 - Step 1: Prepare & Split Dataset
==========================================
Script ini akan:
1. Memvalidasi bahwa setiap gambar memiliki label yang sesuai
2. Membagi dataset menjadi train (80%) dan val (20%)
3. Membuat data.yaml yang benar untuk YOLO11
4. Membuat train.txt dan val.txt yang berisi path gambar

Jalankan dari root project:
    python training/1_prepare_dataset.py
"""

import os
import random
import shutil
from pathlib import Path

# ── Konfigurasi ─────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")

# Script ini mencoba beberapa lokasi gambar secara otomatis:
# 1. dataset/data/images/train/   ← path default dari CVAT export
# 2. dataset/images/train/        ← path alternatif
# 3. dataset/images/              ← path flat
CANDIDATE_IMAGE_DIRS = [
    DATASET_DIR / "data" / "images" / "train",
    DATASET_DIR / "images" / "train",
    DATASET_DIR / "images",
]

LABELS_DIR = DATASET_DIR / "data" / "labels" / "train"

VAL_IMAGES_DIR = DATASET_DIR / "data" / "images" / "val"
VAL_LABELS_DIR = DATASET_DIR / "data" / "labels" / "val"

VAL_SPLIT = 0.2  # 20% untuk validasi
RANDOM_SEED = 42

CLASSES = ["item", "price", "total", "qty"]  # sesuai urutan index di data.yaml

# ── Helpers ──────────────────────────────────────────────────────────────────


def validate_dataset(images_dir: Path, labels_dir: Path) -> list[str]:
    """
    Periksa setiap file gambar, pastikan ada label-nya.
    Return list nama stem yang valid (tanpa ekstensi).
    """
    valid = []
    missing_labels = []

    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            valid.append(img_path.stem)
        else:
            missing_labels.append(img_path.stem)

    if missing_labels:
        print(
            f"[WARNING] {len(missing_labels)} gambar tidak memiliki label dan akan dilewati:"
        )
        for name in missing_labels:
            print(f"  - {name}")

    print(f"[INFO] Dataset valid: {len(valid)} gambar")
    return valid


def split_dataset(stems: list[str], val_split: float, seed: int) -> tuple[list, list]:
    random.seed(seed)
    shuffled = stems.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_split))
    return shuffled[:split_idx], shuffled[split_idx:]


def copy_files(
    stems: list[str],
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        # Cari ekstensi gambar (.png atau .jpg)
        for ext in [".png", ".jpg", ".jpeg"]:
            src_img = src_img_dir / (stem + ext)
            if src_img.exists():
                shutil.copy2(src_img, dst_img_dir / src_img.name)
                break

        src_lbl = src_lbl_dir / (stem + ".txt")
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl_dir / src_lbl.name)


def write_txt_list(stems: list[str], src_img_dir: Path, output_path: Path) -> None:
    """Buat file .txt berisi path gambar relatif terhadap root project."""
    lines = []
    for stem in stems:
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = src_img_dir / (stem + ext)
            if img_path.exists():
                # Path relatif dari root project (pakai forward slash)
                lines.append(str(img_path).replace("\\", "/"))
                break
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Ditulis: {output_path} ({len(lines)} entri)")


def write_data_yaml(dataset_dir: Path, classes: list[str]) -> None:
    """Buat data.yaml yang kompatibel dengan Ultralytics YOLO11."""
    abs_path = dataset_dir.resolve()
    content = f"""# YOLO11 Dataset Configuration
# Receipt Intelligence Platform

path: {abs_path.as_posix()}   # absolute path ke root dataset

train: data/images/train   # relatif terhadap 'path'
val:   data/images/val     # relatif terhadap 'path'

nc: {len(classes)}          # jumlah kelas
names: {classes}
"""
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Ditulis: {yaml_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def find_images_dir() -> Path | None:
    """Cari folder gambar secara otomatis dari beberapa kandidat."""
    for candidate in CANDIDATE_IMAGE_DIRS:
        images = list(candidate.glob("*.png")) + list(candidate.glob("*.jpg"))
        if images:
            print(f"[INFO] Folder gambar ditemukan: {candidate} ({len(images)} file)")
            return candidate
    return None


def main():
    print("=" * 55)
    print("  Phase 3 - Step 1: Dataset Preparation & Split")
    print("=" * 55)

    # 1. Cari folder gambar
    IMAGES_DIR = find_images_dir()
    if IMAGES_DIR is None:
        print("[ERROR] Folder gambar tidak ditemukan!")
        print(
            "        Letakkan gambar receipt (.png/.jpg) di salah satu lokasi berikut:"
        )
        for c in CANDIDATE_IMAGE_DIRS:
            print(f"          - {c}")
        print()
        print("        Catatan: file train.txt di dataset/ berisi path:")
        print("          data/images/train/receipt_xxxxx.png")
        print("        Jadi folder yang benar adalah: dataset/data/images/train/")
        return

    valid_stems = validate_dataset(IMAGES_DIR, LABELS_DIR)
    if not valid_stems:
        print("[ERROR] Tidak ada data valid. Proses dihentikan.")
        return

    # 2. Split
    train_stems, val_stems = split_dataset(valid_stems, VAL_SPLIT, RANDOM_SEED)
    print(f"[INFO] Train: {len(train_stems)} | Val: {len(val_stems)}")

    # 3. Salin file val ke folder val (train sudah di tempatnya)
    print("[INFO] Menyalin file validasi...")
    copy_files(val_stems, IMAGES_DIR, LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR)

    # 4. Buat train.txt & val.txt
    write_txt_list(train_stems, IMAGES_DIR, DATASET_DIR / "train.txt")
    write_txt_list(val_stems, VAL_IMAGES_DIR, DATASET_DIR / "val.txt")

    # 5. Buat data.yaml
    write_data_yaml(DATASET_DIR, CLASSES)

    print()
    print("[DONE] Dataset siap untuk training!")
    print(f"       Gunakan: dataset/data.yaml")


if __name__ == "__main__":
    main()
