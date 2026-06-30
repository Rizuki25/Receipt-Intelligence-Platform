# Phase 3 - YOLO11 Training: Panduan Penggunaan

## Struktur Folder

```
Struk-Monitoring-AI/
├── dataset/
│   ├── data.yaml               ← konfigurasi dataset (auto-generated)
│   ├── train.txt               ← list path gambar train
│   ├── val.txt                 ← list path gambar val (auto-generated)
│   ├── data/
│   │   └── images/
│   │       ├── train/          ← gambar training (receipt_00000.png, dst)
│   │       └── val/            ← gambar validasi (auto-copied)
│   └── labels/
│       ├── train/              ← label YOLO training (.txt)
│       └── val/                ← label YOLO validasi (auto-copied)
├── training/
│   ├── 1_prepare_dataset.py    ← split & validasi dataset
│   ├── 2_train.py              ← training YOLO11
│   ├── 3_evaluate.py           ← evaluasi model
│   └── 4_inference.py          ← test model pada gambar baru
├── runs/                       ← output training (auto-generated)
│   ├── train/receipt_yolo11/
│   │   └── weights/
│   │       ├── best.pt         ← model terbaik
│   │       └── last.pt         ← checkpoint terakhir
│   ├── val/
│   └── inference/
├── requirements.txt
└── workflow.md
```

## Setup

### 1. Buat virtual environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
# Untuk CPU saja:
pip install -r requirements.txt

# Untuk GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Untuk GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Urutan Menjalankan

### Step 1 - Siapkan & split dataset

```bash
python training/1_prepare_dataset.py
```

Output:
- `dataset/data/images/val/` — gambar validasi (20% dari total)
- `dataset/labels/val/` — label validasi
- `dataset/val.txt` — list path gambar val
- `dataset/data.yaml` — konfigurasi final

### Step 2 - Training YOLO11

```bash
# Menggunakan default (yolo11n, 100 epoch, batch 8)
python training/2_train.py

# Custom:
python training/2_train.py --model yolo11s.pt --epochs 150 --batch 16

# Lanjutkan training yang terputus:
python training/2_train.py --resume
```

**Pilihan model YOLO11:**
| Model     | Size   | mAP  | Speed  | Cocok untuk        |
|-----------|--------|------|--------|--------------------|
| yolo11n   | 2.6MB  | baik | cepat  | CPU / GPU rendah   |
| yolo11s   | 9.4MB  | lebih baik | sedang | GPU entry-level |
| yolo11m   | 20MB   | bagus | sedang | GPU mid-range   |
| yolo11l   | 25MB   | sangat bagus | lambat | GPU high-end |

### Step 3 - Evaluasi model

```bash
python training/3_evaluate.py
```

Output metrik:
- **mAP@0.50** — target > 0.80
- **Precision** — target > 0.85
- **Recall** — target > 0.80
- **F1 Score** — target > 0.82

### Step 4 - Inference / Test

```bash
# Test pada satu gambar
python training/4_inference.py --source path/ke/struk.jpg

# Test pada folder
python training/4_inference.py --source dataset/data/images/val/

# Tampilkan hasil di layar
python training/4_inference.py --source struk.jpg --show
```

## Tips

- Jika GPU VRAM kecil (< 4GB), gunakan `--batch 4` atau `--batch 2`
- Jika training terlalu lambat di CPU, kurangi `--epochs 50` untuk percobaan awal
- Hasil training tersimpan otomatis di `runs/train/receipt_yolo11/`
- Gunakan `best.pt` (bukan `last.pt`) untuk inference dan produksi
