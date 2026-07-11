"""
Phase 3 - Step 4: Inference / Test pada Gambar Baru
=====================================================
Script ini menguji model YOLO11 hasil training pada satu atau
beberapa gambar receipt, lalu menyimpan hasilnya.

Jalankan dari root project:
    # Test pada satu gambar
    python training/4_inference.py --source path/ke/struk.jpg

    # Test pada seluruh folder
    python training/4_inference.py --source path/ke/folder/

    # Test pada validation set
    python training/4_inference.py --source dataset/data/images/val/

Filter header:
    --header_ratio  Persentase tinggi gambar yang dianggap area header (default: 0.20)
                    Semua bbox `item` yang center-Y-nya berada di bawah threshold ini
                    akan dibuang, karena nama toko selalu ada di bagian atas struk.
                    Naikkan nilai ini jika header struk lebih panjang.
"""

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ── Konfigurasi Default ───────────────────────────────────────────────────────

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
OUTPUT_DIR = Path("runs/inference")

CLASS_NAMES = {0: "item", 1: "price", 2: "total", 3: "qty"}

# Warna per kelas untuk visualisasi (BGR)
CLASS_COLORS = {
    "item": (255, 100, 50),  # biru
    "price": (50, 200, 50),  # hijau
    "total": (50, 50, 220),  # merah
    "qty": (50, 200, 220),  # kuning
}

# Rasio tinggi gambar yang dianggap area header struk (nama toko, alamat, dsb.)
# Bbox `item` yang center-Y-nya <= threshold ini akan difilter.
DEFAULT_HEADER_RATIO = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────


def find_best_model() -> Path | None:
    """Pilih checkpoint best.pt terbaru agar hasil fine-tuning yang digunakan."""
    if not Path("runs").exists():
        return None
    candidates = list(Path("runs").rglob("best.pt"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def filter_header_items(result, header_ratio: float):
    """
    Filter bbox kelas `item` yang berada di area header struk.

    Logika:
    - Hitung threshold_y = tinggi gambar * header_ratio
    - Jika center-Y dari bbox `item` <= threshold_y → buang (itu nama toko/header)
    - Kelas lain (price, qty, total) tidak difilter

    Return: (boxes_filtered, mask) di mana mask adalah boolean tensor
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return boxes, None

    img_h = result.orig_shape[0]  # tinggi gambar asli (pixel)
    threshold_y = img_h * header_ratio

    keep_mask = []
    filtered_count = 0

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        cls_name = CLASS_NAMES.get(cls_id, "")

        if cls_name == "item":
            # xyxy: [x1, y1, x2, y2]
            y1 = boxes.xyxy[i][1].item()
            y2 = boxes.xyxy[i][3].item()
            center_y = (y1 + y2) / 2

            if center_y <= threshold_y:
                keep_mask.append(False)
                filtered_count += 1
                continue

        keep_mask.append(True)

    return keep_mask, filtered_count


def draw_filtered_result(result, keep_mask: list[bool], output_path: Path) -> None:
    """Gambar ulang bounding box pada gambar asli dengan filter yang sudah diterapkan."""
    img = result.orig_img.copy()
    boxes = result.boxes

    for i, keep in enumerate(keep_mask):
        if not keep:
            continue

        cls_id = int(boxes.cls[i].item())
        conf = boxes.conf[i].item()
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]

        # Gambar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label teks
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def result_output_path(out_dir: Path, image_name: str) -> Path:
    """Buat nama output yang berbeda dari gambar sumber."""
    source_name = Path(image_name)
    return out_dir / f"{source_name.stem}_detected{source_name.suffix}"


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    found_model = find_best_model()
    default_model = (
        str(found_model)
        if found_model
        else "runs/detect/receipt_yolo11/weights/best.pt"
    )

    parser = argparse.ArgumentParser(description="YOLO11 Receipt Inference")
    parser.add_argument("--model", default=default_model, help="Path ke model .pt")
    parser.add_argument("--source", required=True, help="Path gambar atau folder")
    parser.add_argument("--conf", default=CONF_THRESHOLD, type=float)
    parser.add_argument("--iou", default=IOU_THRESHOLD, type=float)
    parser.add_argument("--imgsz", default=IMG_SIZE, type=int)
    parser.add_argument("--device", default="", help="cuda device atau cpu")
    parser.add_argument("--show", action="store_true", help="Tampilkan hasil di layar")
    parser.add_argument(
        "--header_ratio",
        default=DEFAULT_HEADER_RATIO,
        type=float,
        help="Rasio tinggi header struk yang difilter (default: 0.20 = 20%% atas gambar)",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Matikan filter header, tampilkan semua deteksi mentah",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        print("        Jalankan dulu: python training/2_train.py")
        return

    if not source_path.exists():
        print(f"[ERROR] Source tidak ditemukan: {source_path}")
        return

    if args.no_filter:
        print("[INFO] Filter header DINONAKTIFKAN — menampilkan semua deteksi mentah")
    else:
        print(
            f"[INFO] Filter header aktif — area {args.header_ratio * 100:.0f}% atas gambar diabaikan untuk kelas 'item'"
        )

    print(f"[INFO] Memuat model: {model_path}")
    model = YOLO(str(model_path))

    print(f"[INFO] Menjalankan inference pada: {source_path}")
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        save=False,  # kita handle save sendiri setelah filter
        verbose=True,
    )

    # Output folder
    out_dir = OUTPUT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tampilkan ringkasan deteksi
    print()
    print("=" * 55)
    print("  Ringkasan Deteksi")
    print("=" * 55)

    for i, result in enumerate(results):
        img_name = Path(result.path).name
        boxes = result.boxes

        print(f"\n  Gambar {i + 1}: {img_name}")

        if boxes is None or len(boxes) == 0:
            print("    Tidak ada objek terdeteksi.")
            # Simpan gambar asli tanpa anotasi
            cv2.imwrite(str(result_output_path(out_dir, img_name)), result.orig_img)
            continue

        # Terapkan filter header
        if args.no_filter:
            keep_mask = [True] * len(boxes)
            filtered_count = 0
        else:
            keep_mask, filtered_count = filter_header_items(result, args.header_ratio)

        if filtered_count:
            print(f"    [FILTER] {filtered_count} bbox 'item' di area header dihapus")

        # Hitung deteksi per kelas setelah filter
        class_count = {}
        for j, keep in enumerate(keep_mask):
            if not keep:
                continue
            cls_id = int(boxes.cls[j].item())
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            class_count[cls_name] = class_count.get(cls_name, 0) + 1

        if not class_count:
            print("    Tidak ada objek terdeteksi setelah filter.")
        else:
            for cls_name, count in sorted(class_count.items()):
                print(f"    {cls_name:<10}: {count} deteksi")

        # Simpan gambar dengan bbox yang sudah difilter
        # Gunakan nama berbeda agar gambar sumber tidak pernah tertimpa. Ini juga
        # mencegah inference berikutnya membaca bbox yang sudah digambar.
        out_path = result_output_path(out_dir, img_name)
        draw_filtered_result(result, keep_mask, out_path)

        if args.show:
            cv2.imshow(img_name, cv2.imread(str(out_path)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print()
    print(f"  Hasil tersimpan di: {out_dir}")
    print(f"  Gunakan --no_filter untuk melihat deteksi mentah tanpa filter")
    print(f"  Gunakan --header_ratio 0.30 jika header struk lebih panjang")
    print("=" * 55)


if __name__ == "__main__":
    main()
