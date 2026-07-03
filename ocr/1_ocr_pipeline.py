"""
Phase 4 - OCR Pipeline
========================
Menggabungkan YOLO11 (deteksi area) + PaddleOCR (baca teks).
"""

import os

# Atasi konflik OpenMP antara PyTorch dan PaddlePaddle di Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ── Konfigurasi ───────────────────────────────────────────────────────────────

RUN_NAME = "receipt_yolo11"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
HEADER_RATIO = 0.20  # filter area header (nama toko)

OUTPUT_DIR = Path("runs/ocr")

CLASS_NAMES = {0: "item", 1: "price", 2: "total", 3: "qty"}

# Padding di sekitar bbox saat crop (dalam pixel) — beri sedikit ruang agar teks tidak kepotong
CROP_PADDING = 3


# ── Helpers ───────────────────────────────────────────────────────────────────


def find_best_model(run_name: str) -> Path | None:
    """Cari best.pt secara dinamis di seluruh subfolder runs/."""
    if not Path("runs").exists():
        return None
    candidates = list(Path("runs").rglob(f"{run_name}/weights/best.pt"))
    if candidates:
        return candidates[0]
    candidates = list(Path("runs").rglob("best.pt"))
    if candidates:
        return candidates[0]
    return None


def crop_bbox(
    img: np.ndarray, xyxy: list[float], padding: int = CROP_PADDING
) -> np.ndarray:
    """Crop area bbox dari gambar dengan sedikit padding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return img[y1:y2, x1:x2]


def read_text_from_crop(ocr: PaddleOCR, crop: np.ndarray) -> str:
    """Jalankan OCR pada gambar hasil crop, return teks yang terbaca."""
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return ""

    try:
        # Jika gambar crop sangat sempit/kecil (seperti qty), matikan deteksi kotak teks (det=False).
        # Ini memaksa PaddleOCR langsung mengenali karakter (recognition) tanpa mencari kotak paragraf lagi.
        is_small = crop.shape[0] < 30 or crop.shape[1] < 50

        if is_small:
            # Mode Recognition saja
            result = ocr.ocr(crop, det=False, rec=True, cls=False)
        else:
            # Mode standard (deteksi baris dulu baru baca)
            result = ocr.ocr(crop, det=True, rec=True, cls=False)

    except Exception as e:
        print(f"    [WARN] OCR error: {e}")
        return ""

    if not result:
        return ""

    # Parse hasil
    texts = []

    # Format output det=False: [('text', confidence)]
    # Format output det=True : [[ [bbox], ('text', confidence) ], ...]
    if is_small:
        # result berbentuk: [('text', confidence)] atau [[('text', confidence)]]
        if isinstance(result[0], tuple):
            texts.append(str(result[0][0]))
        elif isinstance(result[0], list) and len(result[0]) > 0:
            if isinstance(result[0][0], tuple):
                texts.append(str(result[0][0][0]))
    else:
        # result berbentuk: [[ [bbox], ('text', confidence) ], ...]
        if result[0] is not None:
            for line in result[0]:
                if line and len(line) >= 2:
                    text_part = line[1]
                    if isinstance(text_part, (list, tuple)) and len(text_part) >= 1:
                        texts.append(str(text_part[0]))

    return " ".join(t.strip() for t in texts if t and t.strip())


def clean_number(text: str) -> int | None:
    """Bersihkan teks harga/qty, konversi ke integer. Return None jika gagal."""
    if not text:
        return None
    # Hapus semua karakter non-digit
    digits = "".join(c for c in text if c.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def process_image(
    model: YOLO,
    ocr: PaddleOCR,
    img_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    header_ratio: float,
) -> dict:
    """
    Proses satu gambar: deteksi + OCR + filter header.
    Return dict dengan hasil terstruktur.
    """
    print(f"\n[PROCESS] {img_path.name}")

    # Baca gambar
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"    [ERROR] Gagal baca gambar: {img_path}")
        return {"image": img_path.name, "error": "cannot read image"}

    img_h = img.shape[0]
    header_y = img_h * header_ratio

    # YOLO detection
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("    [WARN] Tidak ada objek terdeteksi")
        return {
            "image": img_path.name,
            "items": [],
            "total": None,
            "raw_detections": [],
        }

    boxes = results[0].boxes

    # Kumpulkan semua deteksi + terapkan filter header + OCR
    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        conf_val = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        x1, y1, x2, y2 = xyxy
        center_y = (y1 + y2) / 2

        # Filter header untuk kelas 'item'
        if cls_name == "item" and center_y <= header_y:
            continue

        # Crop + OCR
        crop = crop_bbox(img, xyxy)
        text = read_text_from_crop(ocr, crop)

        detections.append(
            {
                "class": cls_name,
                "conf": round(conf_val, 3),
                "bbox": [round(v, 1) for v in xyxy],
                "center_y": center_y,
                "text": text,
            }
        )

    # Susun jadi struktur item/qty/price + total
    print("    [DEBUG] Deteksi Mentah:")
    for d in detections:
        print(
            f"      - Class: {d['class']:<6} | Conf: {d['conf']:.2f} | Y: {d['center_y']:.1f} | OCR Text: '{d['text']}'"
        )

    structured = organize_detections(
        detections, img=img, ocr=ocr, header_ratio=header_ratio
    )

    print(f"    Items terdeteksi: {len(structured['items'])}")
    print(f"    Total           : {structured['total']}")

    return {
        "image": img_path.name,
        "items": structured["items"],
        "total": structured["total"],
        "raw_detections": detections,
    }


def find_total_by_keyword(img: np.ndarray, ocr: PaddleOCR) -> int | None:
    """
    Fallback method: cari total belanja menggunakan OCR berbasis kata kunci teks.
    Hanya dijalankan jika YOLO gagal mendeteksi total.
    Kita batasi OCR hanya pada 30% area bawah gambar struk untuk efisiensi dan keamanan.
    """
    h, w = img.shape[:2]
    # Potong 30% bagian bawah gambar struk
    crop_bottom = img[int(h * 0.65) :, :]

    try:
        # Jalankan full OCR pada area bawah struk saja
        result = ocr.ocr(crop_bottom, det=True, rec=True, cls=False)
    except Exception as e:
        print(f"    [WARN] Fallback OCR error: {e}")
        return None

    if not result or result[0] is None:
        return None

    # Ubah ke list line yang diurutkan vertikal ke horisontal
    lines = []
    for line in result[0]:
        if line and len(line) >= 2:
            bbox, (text, conf) = line
            text_upper = text.upper().strip()
            # Gunakan center-Y relatif terhadap crop_bottom untuk sorting
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            center_x = (bbox[0][0] + bbox[2][0]) / 2
            lines.append({"text": text_upper, "y": center_y, "x": center_x})

    # Urutkan lines dari atas ke bawah
    lines.sort(key=lambda l: l["y"])

    # Kata kunci penanda total belanja
    keywords = [
        "SUBTOTAL",
        "SUB TOTAL",
        "TOTAL",
        "DUE",
        "GRAND TOTAL",
        "BELANJA",
        "TOTAL BELANJA",
    ]

    # Cari line yang mengandung kata kunci
    for i, line in enumerate(lines):
        text = line["text"]
        if any(kw in text for kw in keywords):
            # Cek kasus 1: Nilai angka ada di line yang sama (misal: 'TOTAL 24,500' atau 'DUE: 24.500')
            val = clean_number(text)
            if (
                val is not None and val > 100
            ):  # abaikan angka kecil yang mungkin bukan total
                return val

            # Cek kasus 2: Nilai angka berada di baris yang sama secara horizontal tapi beda bbox
            # Cari baris lain yang koordinat Y-nya sangat mirip (+- 15 pixel) tapi posisinya lebih ke kanan (X > X_key)
            matched_horizontal = []
            for other in lines:
                if (
                    other != line
                    and abs(other["y"] - line["y"]) <= 20
                    and other["x"] > line["x"]
                ):
                    val = clean_number(other["text"])
                    if val is not None:
                        matched_horizontal.append((other["x"], val))
            if matched_horizontal:
                # Ambil yang posisinya paling kanan
                matched_horizontal.sort(key=lambda x: x[0], reverse=True)
                return matched_horizontal[0][1]

            # Cek kasus 3: Angka berada di baris tepat di bawahnya (biasanya di struk tertentu)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                val = clean_number(next_line["text"])
                if val is not None and val > 100:
                    return val

    return None


def organize_detections(
    detections: list[dict],
    img: np.ndarray = None,
    ocr: PaddleOCR = None,
    header_ratio: float = 0.20,
) -> dict:
    """
    Kelompokkan deteksi berdasarkan baris (Y position).
    Item, qty, dan price di baris yang sama biasanya adalah satu produk.
    """
    # 1. Tentukan toleransi Y dinamis berdasarkan tinggi bounding box rata-rata
    box_heights = [d["bbox"][3] - d["bbox"][1] for d in detections]
    Y_TOLERANCE = max(30, int(np.mean(box_heights) * 1.2)) if box_heights else 35

    h_img, w_img = img.shape[:2] if img is not None else (1000, 1000)
    header_y = h_img * header_ratio

    # 2. Jika YOLO mendeteksi price/qty tapi melewatkan item (low confidence),
    #    gunakan fallback OCR pada area sebelah kiri price/qty untuk merekonstruksi item.
    if img is not None and ocr is not None:
        init_items = [d for d in detections if d["class"] == "item"]
        prices = [d for d in detections if d["class"] == "price"]
        qtys = [d for d in detections if d["class"] == "qty"]
        added_items = []

        # Cari fallback untuk price/qty yang tidak berjodoh dengan item apapun
        for p in sorted(prices + qtys, key=lambda x: x["center_y"]):
            p_ctr_y = p["center_y"]

            # Abaikan jika ada di area header
            if p_ctr_y <= header_y:
                continue

            # Periksa kecocokan vertikal
            has_match = False
            for it in init_items + added_items:
                if abs(it["center_y"] - p_ctr_y) <= Y_TOLERANCE:
                    has_match = True
                    break

            if not has_match:
                # Crop area sebelah kiri price/qty
                x1_p, y1_p, x2_p, y2_p = p["bbox"]
                y_pad = 5
                crop_y1 = max(0, int(y1_p - y_pad))
                crop_y2 = min(h_img, int(y2_p + y_pad))
                crop_x1 = 0
                crop_x2 = max(0, int(x1_p - 10))

                if (crop_x2 - crop_x1) > 50 and (crop_y2 - crop_y1) > 10:
                    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    fallback_text = read_text_from_crop(ocr, crop_img)

                    if fallback_text:
                        fallback_text = fallback_text.strip()
                        alpha_chars = [c for c in fallback_text if c.isalpha()]
                        # Tambahkan jika teks cukup bermakna (mengandung setidaknya 3 huruf)
                        if len(alpha_chars) >= 3:
                            new_item = {
                                "class": "item",
                                "conf": 1.0,
                                "bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                                "center_y": (crop_y1 + crop_y2) / 2,
                                "text": fallback_text,
                            }
                            added_items.append(new_item)
                            print(
                                f"    [OCR FALLBACK ITEM] Berhasil mendeteksi item virtual: '{fallback_text}' pada Y: {p_ctr_y:.1f}"
                            )

        # Gabungkan deteksi virtual
        detections.extend(added_items)

    # Pisahkan berdasarkan kelas terbaru
    items = [d for d in detections if d["class"] == "item"]
    qtys = [d for d in detections if d["class"] == "qty"]
    prices = [d for d in detections if d["class"] == "price"]
    totals = [d for d in detections if d["class"] == "total"]

    # Urutkan item berdasarkan posisi vertikal
    items.sort(key=lambda d: d["center_y"])

    structured_items = []
    for item in items:
        item_y = item["center_y"]

        # Cari qty & price di baris yang sama (Y berdekatan)
        matched_qty = min(
            (q for q in qtys if abs(q["center_y"] - item_y) <= Y_TOLERANCE),
            key=lambda q: abs(q["center_y"] - item_y),
            default=None,
        )
        matched_price = min(
            (p for p in prices if abs(p["center_y"] - item_y) <= Y_TOLERANCE),
            key=lambda p: abs(p["center_y"] - item_y),
            default=None,
        )

        structured_items.append(
            {
                "name": item["text"],
                "qty": clean_number(matched_qty["text"]) if matched_qty else None,
                "price": clean_number(matched_price["text"]) if matched_price else None,
            }
        )

    # Total: ambil dari YOLO jika terdeteksi
    total_value = None
    if totals:
        best_total = max(totals, key=lambda d: d["conf"])
        total_value = clean_number(best_total["text"])

    # JIKA YOLO gagal mendeteksi total, aktifkan fallback OCR pintar
    if total_value is None and img is not None and ocr is not None:
        print(
            "    [OCR FALLBACK] YOLO gagal mendeteksi total. Mencari via kata kunci..."
        )
        total_value = find_total_by_keyword(img, ocr)

    return {"items": structured_items, "total": total_value}


def save_visualization(img_path: Path, result_data: dict, out_dir: Path) -> None:
    """Gambar bbox + teks OCR pada gambar asli untuk verifikasi visual."""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    colors = {
        "item": (255, 100, 50),
        "price": (50, 200, 50),
        "total": (50, 50, 220),
        "qty": (50, 200, 220),
    }

    for det in result_data.get("raw_detections", []):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = colors.get(det["class"], (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label: kelas + teks OCR (dipotong biar tidak kepanjangan)
        text_preview = det["text"][:20] + ("..." if len(det["text"]) > 20 else "")
        label = f"{det['class']}: {text_preview}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), img)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    found_model = find_best_model(RUN_NAME)
    default_model = (
        str(found_model)
        if found_model
        else "runs/detect/receipt_yolo11/weights/best.pt"
    )

    parser = argparse.ArgumentParser(description="YOLO + PaddleOCR Pipeline")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--source", required=True, help="Gambar atau folder")
    parser.add_argument("--conf", default=CONF_THRESHOLD, type=float)
    parser.add_argument("--iou", default=IOU_THRESHOLD, type=float)
    parser.add_argument("--imgsz", default=IMG_SIZE, type=int)
    parser.add_argument("--header_ratio", default=HEADER_RATIO, type=float)
    parser.add_argument("--lang", default="en", help="Bahasa OCR (en/id/ch)")
    parser.add_argument("--save_json", action="store_true", help="Simpan hasil ke JSON")
    parser.add_argument(
        "--no_visualize", action="store_true", help="Jangan simpan gambar visualisasi"
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    """Kumpulkan semua gambar dari source (file atau folder)."""
    if source.is_file():
        return [source]

    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in extensions)


def main():
    args = parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return

    if not source_path.exists():
        print(f"[ERROR] Source tidak ditemukan: {source_path}")
        return

    images = collect_images(source_path)
    if not images:
        print(f"[ERROR] Tidak ada gambar di: {source_path}")
        return

    print("=" * 55)
    print(f"  Phase 4 - OCR Pipeline")
    print("=" * 55)
    print(f"  Model  : {model_path}")
    print(f"  Source : {source_path} ({len(images)} gambar)")
    print(f"  Lang   : {args.lang}")
    print("=" * 55)

    print("[INFO] Memuat YOLO model...")
    model = YOLO(str(model_path))

    print("[INFO] Memuat PaddleOCR (bisa memakan waktu saat pertama kali)...")
    ocr = PaddleOCR(lang=args.lang, use_textline_orientation=False)

    out_dir = OUTPUT_DIR / "results"
    vis_dir = out_dir / "visualizations"
    json_dir = out_dir / "json"

    all_results = []

    for img_path in images:
        result = process_image(
            model,
            ocr,
            img_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            header_ratio=args.header_ratio,
        )
        all_results.append(result)

        # Visualisasi
        if not args.no_visualize:
            save_visualization(img_path, result, vis_dir)

        # Simpan JSON per gambar
        if args.save_json:
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"{img_path.stem}.json"
            # Buang raw_detections dari JSON output final agar lebih rapi
            clean_result = {
                "image": result["image"],
                "items": result["items"],
                "total": result["total"],
            }
            json_path.write_text(
                json.dumps(clean_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # Ringkasan
    print()
    print("=" * 55)
    print("  Ringkasan")
    print("=" * 55)

    for r in all_results:
        print(f"\n  {r['image']}")
        if "error" in r:
            print(f"    [ERROR] {r['error']}")
            continue

        for item in r["items"]:
            name = item["name"] or "(?)"
            qty = item["qty"] if item["qty"] is not None else "-"
            price = item["price"] if item["price"] is not None else "-"
            print(f"    • {name:<30} qty={qty:<4} price={price}")

        total = r["total"] if r["total"] is not None else "(tidak terdeteksi)"
        print(f"    TOTAL: {total}")

    print()
    if not args.no_visualize:
        print(f"  Visualisasi: {vis_dir}")
    if args.save_json:
        print(f"  JSON       : {json_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
