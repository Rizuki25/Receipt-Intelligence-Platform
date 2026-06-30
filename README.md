# 🧾 Receipt Intelligence Platform

Aplikasi berbasis AI untuk membaca struk belanja secara otomatis menggunakan **Computer Vision (YOLO11)** dan **OCR (PaddleOCR)**, kemudian menyimpan hasil ekstraksi ke database dan menampilkan analisis pengeluaran dalam dashboard interaktif.

---

## ✨ Fitur Utama

- 📷 **Auto-detect** area item, qty, harga, dan total dari foto struk
- 🔤 **OCR otomatis** untuk membaca teks dari setiap area yang terdeteksi
- 🏷️ **Kategorisasi AI** — item otomatis dikategorikan (Makanan, Minuman, dll.)
- 📊 **Dashboard analitik** — grafik pengeluaran bulanan & mingguan
- ✏️ **Koreksi manual** — pengguna dapat mengedit hasil OCR jika ada kesalahan

---

## 🛠️ Technology Stack

| Layer | Teknologi |
|---|---|
| Frontend | React, Tailwind CSS |
| Backend | FastAPI / Flask |
| Computer Vision | Ultralytics YOLO11 |
| OCR | PaddleOCR |
| Database | PostgreSQL / SQLite |
| Visualisasi | Chart.js / Recharts |

---

## 🔄 Pipeline

```
Upload Struk → Input Tanggal → YOLO Detection → Crop Objek
     → PaddleOCR → Parsing Data → Kategorisasi AI → Database → Dashboard
```

---

## 📁 Struktur Proyek

```
Struk-Monitoring-AI/
├── dataset/
│   ├── data.yaml               # Konfigurasi dataset YOLO
│   ├── train.txt               # List path gambar train
│   ├── val.txt                 # List path gambar val
│   ├── data/
│   │   └── images/
│   │       ├── train/          # Gambar training
│   │       └── val/            # Gambar validasi
│   └── labels/
│       ├── train/              # Label YOLO training (.txt)
│       └── val/                # Label YOLO validasi (.txt)
├── training/
│   ├── 1_prepare_dataset.py    # Split & validasi dataset
│   ├── 2_train.py              # Training YOLO11
│   ├── 3_evaluate.py           # Evaluasi model
│   └── 4_inference.py          # Test model pada gambar baru
├── runs/                       # Output training (auto-generated)
│   └── train/receipt_yolo11/
│       └── weights/
│           ├── best.pt         # Model terbaik
│           └── last.pt         # Checkpoint terakhir
├── requirements.txt
├── workflow.md
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/Struk-Monitoring-AI.git
cd Struk-Monitoring-AI
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# PyYAML
pip install "pyyaml>=6.0"

# PyTorch (CPU only)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# Ultralytics YOLO11
pip install "ultralytics>=8.3.0"

# Atau install semua sekaligus (pastikan torch sudah terinstall duluan)
pip install -r requirements.txt
```

### 4. Siapkan Dataset

```bash
python training/1_prepare_dataset.py
```

Script ini akan:
- Memvalidasi pasangan gambar & label
- Membagi dataset menjadi **train (80%)** dan **val (20%)**
- Membuat `dataset/data.yaml` yang siap digunakan

### 5. Training Model

```bash
# Default (yolo11n, 100 epoch, batch 8)
python training/2_train.py

# Kustom
python training/2_train.py --model yolo11s.pt --epochs 150 --batch 4

# Lanjutkan training yang terputus
python training/2_train.py --resume
```

### 6. Evaluasi Model

```bash
python training/3_evaluate.py
```

Output metrik yang dihasilkan:

| Metrik | Target |
|---|---|
| mAP@0.50 | > 0.80 |
| Precision | > 0.85 |
| Recall | > 0.80 |
| F1 Score | > 0.82 |

### 7. Inference / Test

```bash
# Test pada satu gambar
python training/4_inference.py --source path/ke/struk.jpg

# Test pada seluruh folder
python training/4_inference.py --source dataset/data/images/val/
```

---

## 🏷️ Label Deteksi

Model YOLO11 dilatih untuk mendeteksi 4 kelas pada struk belanja:

| ID | Label | Deskripsi |
|---|---|---|
| 0 | `item` | Nama produk / barang |
| 1 | `price` | Harga per item |
| 2 | `total` | Total pembayaran |
| 3 | `qty` | Jumlah / kuantitas item |

---

## 📊 Output Parsing

Setelah YOLO + OCR, data dikonversi menjadi format terstruktur:

```json
{
  "date": "2024-01-15",
  "items": [
    {
      "name": "REAL GANACHE",
      "qty": 1,
      "price": 16500,
      "category": "Makanan"
    },
    {
      "name": "ICED HIBISCUS LYCHEE TEA",
      "qty": 1,
      "price": 37000,
      "category": "Minuman"
    }
  ],
  "total": 57500
}
```

---

## 🗺️ Roadmap

- [x] Dataset preparation & annotation
- [x] YOLO11 training pipeline
- [x] Model evaluation
- [x] Inference script
- [ ] OCR integration (PaddleOCR)
- [ ] Data parsing
- [ ] Database (PostgreSQL / SQLite)
- [ ] Backend API (FastAPI / Flask)
- [ ] Frontend upload & dashboard (React)
- [ ] AI categorization
- [ ] Testing & deployment

---

## 📄 Dataset

Menggunakan **Indonesian CORD Receipt Dataset** yang telah dianotasi ulang menggunakan [CVAT](https://cvat.ai/) dengan 4 label khusus: `item`, `qty`, `price`, `total`.

> ⚠️ Tanggal transaksi tidak digunakan karena pada dataset CORD sebagian besar sudah diblur. Tanggal pembelian diinput manual oleh pengguna.

---

## 📝 License

This project is licensed under the MIT License.
