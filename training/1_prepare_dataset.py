"""
Phase 3 - Step 1: Prepare & Split Dataset
==========================================

Script ini aman dijalankan ulang. Semua pasangan gambar/label yang berada di
folder train dan val akan dikumpulkan, divalidasi, lalu dibagi ulang menjadi
train (80%) dan val (20%) tanpa data yang tumpang tindih.

Jalankan dari root project:
    python training/1_prepare_dataset.py
"""

import random
import shutil
import tempfile
from pathlib import Path


DATASET_DIR = Path("dataset")
DATA_DIR = DATASET_DIR / "data"

TRAIN_IMAGES_DIR = DATA_DIR / "images" / "train"
VAL_IMAGES_DIR = DATA_DIR / "images" / "val"
TRAIN_LABELS_DIR = DATA_DIR / "labels" / "train"
VAL_LABELS_DIR = DATA_DIR / "labels" / "val"
UNLABELED_IMAGES_DIR = DATA_DIR / "unlabeled" / "images"

VAL_SPLIT = 0.2
RANDOM_SEED = 42
CLASSES = ["item", "price", "total", "qty"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_unique_files(directories: list[Path], extensions: set[str]) -> dict[str, Path]:
    """Kumpulkan file unik berdasarkan stem dari beberapa folder."""
    files: dict[str, Path] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                existing = files.get(path.stem)
                if existing and existing.read_bytes() != path.read_bytes():
                    raise ValueError(
                        f"Nama file bentrok tetapi isinya berbeda: {existing} dan {path}"
                    )
                files[path.stem] = path
    return files


def validate_label(label_path: Path) -> None:
    """Validasi format baris YOLO: class x_center y_center width height."""
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Format label tidak valid: {label_path}:{line_number}")
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"Nilai label bukan angka: {label_path}:{line_number}"
            ) from exc
        if class_id not in range(len(CLASSES)):
            raise ValueError(f"Class ID di luar rentang: {label_path}:{line_number}")
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(
                f"Koordinat harus berada pada rentang 0-1: {label_path}:{line_number}"
            )


def clear_dataset_files(directory: Path, extensions: set[str]) -> None:
    """Hapus hanya file dataset yang akan dibuat ulang, bukan folder lain."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            path.unlink()


def write_txt_list(image_paths: list[Path], output_path: Path) -> None:
    lines = [path.as_posix() for path in image_paths]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Ditulis: {output_path} ({len(lines)} entri)")


def write_data_yaml() -> None:
    content = f"""# YOLO11 Dataset Configuration
# Receipt Intelligence Platform

path: {DATASET_DIR.resolve().as_posix()}

train: data/images/train
val: data/images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Ditulis: {yaml_path}")


def main() -> None:
    print("=" * 55)
    print("  Phase 3 - Step 1: Dataset Preparation & Split")
    print("=" * 55)

    images = collect_unique_files(
        [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR], IMAGE_EXTENSIONS
    )
    labels = collect_unique_files(
        [TRAIN_LABELS_DIR, VAL_LABELS_DIR], {".txt"}
    )

    if not images:
        print("[ERROR] Tidak ada gambar di folder train atau val.")
        return

    orphan_labels = sorted(set(labels) - set(images))
    if orphan_labels:
        raise ValueError(
            "Label tanpa gambar ditemukan: " + ", ".join(orphan_labels)
        )

    valid_stems = sorted(set(images) & set(labels))
    unlabeled_stems = sorted(set(images) - set(labels))
    for stem in valid_stems:
        validate_label(labels[stem])

    if not valid_stems:
        print("[ERROR] Tidak ada pasangan gambar dan label yang valid.")
        return

    shuffled = valid_stems.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * VAL_SPLIT))
    val_stems = set(shuffled[-val_count:])
    train_stems = [stem for stem in shuffled if stem not in val_stems]

    # Simpan salinan sementara sebelum folder train/val dibersihkan.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare_", dir=DATA_DIR) as temp_name:
        temp_dir = Path(temp_name)
        temp_images = temp_dir / "images"
        temp_labels = temp_dir / "labels"
        temp_images.mkdir()
        temp_labels.mkdir()

        for stem, path in images.items():
            shutil.copy2(path, temp_images / path.name)
        for stem, path in labels.items():
            shutil.copy2(path, temp_labels / f"{stem}.txt")

        for directory, extensions in [
            (TRAIN_IMAGES_DIR, IMAGE_EXTENSIONS),
            (VAL_IMAGES_DIR, IMAGE_EXTENSIONS),
            (TRAIN_LABELS_DIR, {".txt"}),
            (VAL_LABELS_DIR, {".txt"}),
        ]:
            clear_dataset_files(directory, extensions)

        output_images: dict[str, Path] = {}
        for stem in train_stems:
            source_image = next(temp_images.glob(f"{stem}.*"))
            destination = TRAIN_IMAGES_DIR / source_image.name
            shutil.copy2(source_image, destination)
            shutil.copy2(temp_labels / f"{stem}.txt", TRAIN_LABELS_DIR / f"{stem}.txt")
            output_images[stem] = destination

        for stem in sorted(val_stems):
            source_image = next(temp_images.glob(f"{stem}.*"))
            destination = VAL_IMAGES_DIR / source_image.name
            shutil.copy2(source_image, destination)
            shutil.copy2(temp_labels / f"{stem}.txt", VAL_LABELS_DIR / f"{stem}.txt")
            output_images[stem] = destination

        if unlabeled_stems:
            UNLABELED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            for stem in unlabeled_stems:
                source_image = next(temp_images.glob(f"{stem}.*"))
                shutil.copy2(source_image, UNLABELED_IMAGES_DIR / source_image.name)

    train_paths = [output_images[stem] for stem in train_stems]
    val_paths = [output_images[stem] for stem in sorted(val_stems)]
    write_txt_list(train_paths, DATASET_DIR / "train.txt")
    write_txt_list(val_paths, DATASET_DIR / "val.txt")
    write_data_yaml()

    print(f"[INFO] Total berlabel: {len(valid_stems)}")
    print(f"[INFO] Train: {len(train_stems)} | Val: {len(val_stems)}")
    if unlabeled_stems:
        print(
            f"[WARNING] {len(unlabeled_stems)} gambar tanpa label dipindahkan ke "
            f"{UNLABELED_IMAGES_DIR}: {', '.join(unlabeled_stems)}"
        )
    print("[DONE] Dataset siap untuk training tanpa overlap train/val.")


if __name__ == "__main__":
    main()
