# 🎵 Spotify Music Analyzer & Recommender

Aplikasi web berbasis **Streamlit** yang memakai **Machine Learning** untuk
menganalisis selera musik dan memberi rekomendasi lagu berdasarkan
*audio features*.

Aplikasi ini merupakan pengembangan dari notebook analisis
*Spotify Music Recommendation* menjadi sebuah website yang interaktif.

---

## ✨ Fitur Utama

| Halaman | Deskripsi |
|---|---|
| 🏠 **Beranda** | Ringkasan aplikasi & statistik dataset. |
| 📊 **Analisis Data** | EDA interaktif: balance target, distribusi fitur, radar chart, korelasi, PCA, perbandingan model, feature importance, dan confusion matrix. |
| 🤖 **Prediksi & Rekomendasi** | Atur profil audio lagu → prediksi "disukai/tidak" + rekomendasi lagu serupa. |
| ℹ️ **Tentang** | Penjelasan aplikasi & alur machine learning. |

---

## 🗂️ Struktur Project

```
spotify-streamlit-app/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Dependensi Python
├── README.md
├── utils/
│   ├── __init__.py
│   ├── data_processing.py  # Pemuatan data, training model, prediksi, rekomendasi
│   └── visualization.py    # Semua visualisasi (Plotly)
├── assets/
│   └── logo.png
├── data/
│   ├── README.md
│   └── data.csv            # Letakkan dataset Anda di sini
└── .streamlit/
    └── config.toml         # Tema tampilan aplikasi
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Siapkan project
```bash
cd spotify-streamlit-app
```

### 2. Buat virtual environment (disarankan)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Pasang dependensi
```bash
pip install -r requirements.txt
```

### 4. (Opsional) Letakkan dataset Anda
Salin `data.csv` ke folder `data/`. Jika belum ada, aplikasi otomatis memakai
dataset demo sintetis.

### 5. Jalankan
```bash
streamlit run app.py
```
Aplikasi akan terbuka di `http://localhost:8501`.

---

## ☁️ Deployment ke Streamlit Community Cloud

1. Push / unggah project ke GitHub (folder `venv` tidak perlu ikut).
2. Buka <https://share.streamlit.io> → **Create app** → pilih repo & `app.py`.
3. Klik **Deploy**.

> Aplikasi ini tidak memakai kredensial atau API eksternal, jadi tidak perlu
> mengatur *secrets* apa pun saat deployment.

---

## 🧠 Ringkasan Pipeline Machine Learning

Diadaptasi dari notebook asli:

1. **EDA** — cek balance target, distribusi fitur, outlier sebagai sinyal.
2. **Feature selection** — membuang `energy` (korelasi tinggi dengan `loudness`).
3. **Preprocessing** — `train_test_split` 80:20 + `StandardScaler`.
4. **Model** — membandingkan *Logistic Regression*, *SVM*, dan *Random Forest*.
   Random Forest menjadi model terbaik.
5. **Tuning (opsional)** — `GridSearchCV` untuk mengoptimalkan Random Forest.
6. **Rekomendasi** — content-based (jarak Euclidean pada fitur ter-*scale*),
   disaring oleh prediksi model agar hanya merekomendasikan lagu yang
   diprediksi disukai.

---

## 🛠️ Teknologi

- [Streamlit](https://streamlit.io) — antarmuka web
- [scikit-learn](https://scikit-learn.org) — machine learning
- [Plotly](https://plotly.com/python/) — visualisasi interaktif
- [pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org)

---

## 📄 Lisensi

Bebas digunakan untuk keperluan pembelajaran dan pengembangan.
