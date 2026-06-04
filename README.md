# 🎵 Spotify Music Explorer & Recommender

Aplikasi web berbasis **Streamlit** yang menggabungkan **Spotify Web API**
dengan model **Machine Learning** untuk menjelajahi musik, menganalisis selera
lagu, dan memberi rekomendasi berdasarkan *audio features*.

Aplikasi ini merupakan pengembangan dari notebook analisis
*Spotify Music Recommendation* menjadi sebuah website yang interaktif.

---

## ✨ Fitur Utama

| Halaman | Deskripsi |
|---|---|
| 🏠 **Beranda** | Ringkasan aplikasi & statistik dataset. |
| 🔎 **Pencarian Spotify** | Cari lagu, artis, album live dari Spotify API — lengkap dengan popularity, genre, durasi, cover art, dan preview. |
| 📊 **Analisis Data** | EDA interaktif: balance target, distribusi fitur, radar chart, korelasi, PCA, perbandingan model, feature importance, dan confusion matrix. |
| 🤖 **Prediksi & Rekomendasi** | Atur profil audio lagu → prediksi "disukai/tidak" + rekomendasi lagu serupa. |
| ℹ️ **Tentang** | Penjelasan aplikasi & catatan teknis. |

---

## ⚠️ Catatan Teknis Penting (wajib dibaca)

Sejak **27 November 2024**, Spotify **menghentikan (deprecate)** beberapa
endpoint Web API untuk aplikasi **baru**, di antaranya:

- ❌ Audio Features (danceability, energy, valence, dll.)
- ❌ Audio Analysis
- ❌ Recommendations
- ❌ Related Artists

Karena model machine learning pada proyek ini dibangun di atas *audio
features*, sedangkan endpoint tersebut tidak lagi dapat diakses aplikasi baru,
aplikasi ini menggunakan **arsitektur dua sumber data**:

1. **Spotify Web API (live)** → pencarian lagu/artis/album, **popularity**,
   **genre** (dari artis), durasi, dan cover art. (Endpoint ini masih aktif.)
2. **Dataset lokal (`data/data.csv`) + model ML** → analisis (EDA), prediksi
   *liked*, dan rekomendasi berbasis audio features.

Dengan begitu seluruh nilai analitis notebook tetap berfungsi penuh, sementara
bagian yang bergantung pada API tetap memakai data Spotify yang sah.

---

## 🗂️ Struktur Project

```
spotify-streamlit-app/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Dependensi Python
├── .env.example            # Contoh konfigurasi kredensial (lokal)
├── .gitignore
├── README.md
├── utils/
│   ├── __init__.py
│   ├── spotify_api.py      # Autentikasi & pemanggilan Spotify API
│   ├── data_processing.py  # Pemuatan data, training model, prediksi, rekomendasi
│   └── visualization.py    # Semua visualisasi (Plotly)
├── assets/
│   └── logo.png
├── data/
│   ├── README.md
│   └── data.csv            # (letakkan dataset Anda di sini)
└── .streamlit/
    └── secrets.toml.example  # Contoh konfigurasi rahasia untuk deployment
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Kloning / siapkan project
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

### 4. Konfigurasi kredensial Spotify
```bash
cp .env.example .env
```
Lalu buka `.env` dan isi `SPOTIFY_CLIENT_ID` serta `SPOTIFY_CLIENT_SECRET`
(lihat panduan di bawah).

### 5. (Opsional) Letakkan dataset Anda
Salin `data.csv` ke folder `data/`. Jika belum ada, aplikasi otomatis memakai
dataset demo sintetis.

### 6. Jalankan
```bash
streamlit run app.py
```
Aplikasi akan terbuka di `http://localhost:8501`.

---

## 🔑 Cara Mendapatkan Spotify Client ID & Client Secret

1. Buka **Spotify Developer Dashboard**:
   <https://developer.spotify.com/dashboard> dan login dengan akun Spotify
   (akun gratis pun bisa).
2. Klik **Create app**.
3. Isi formulir:
   - **App name** & **App description**: bebas (mis. "Music Explorer").
   - **Redirect URI**: isi `http://127.0.0.1:8501/callback`
     (aplikasi ini memakai *Client Credentials Flow* sehingga redirect URI
     tidak dipakai untuk login, tetapi kolom ini wajib diisi saat membuat app).
   - **Which API/SDKs**: centang **Web API**.
   - Centang persetujuan Developer Terms, lalu **Save**.
4. Masuk ke halaman app → **Settings**.
5. Salin **Client ID**. Untuk **Client Secret**, klik **View client secret**.
6. Tempelkan keduanya ke file `.env` Anda:
   ```env
   SPOTIFY_CLIENT_ID=xxxxxxxxxxxxxxxx
   SPOTIFY_CLIENT_SECRET=xxxxxxxxxxxxxxxx
   ```

> 🔒 **Keamanan:** Client Secret tidak pernah ditulis langsung di dalam kode.
> Aplikasi membacanya dari `st.secrets` (saat deploy) atau dari `.env` (saat
> lokal). File `.env` sudah di-*ignore* oleh Git.

---

## ☁️ Deployment ke Streamlit Community Cloud

1. Push project ke GitHub (pastikan `.env` **tidak** ikut ter-commit).
2. Buka <https://share.streamlit.io> → **New app** → pilih repo & `app.py`.
3. Pada **Advanced settings → Secrets**, tempelkan:
   ```toml
   SPOTIFY_CLIENT_ID = "xxxxxxxxxxxxxxxx"
   SPOTIFY_CLIENT_SECRET = "xxxxxxxxxxxxxxxx"
   ```
4. Klik **Deploy**.

---

## 🧠 Ringkasan Pipeline Machine Learning

Diadaptasi dari notebook asli:

1. **EDA** — cek balance target, distribusi fitur, outlier sebagai sinyal.
2. **Feature selection** — membuang `energy` (korelasi 0.81 dengan `loudness`).
3. **Preprocessing** — `train_test_split` 80:20 + `StandardScaler`.
4. **Model** — membandingkan *Logistic Regression*, *SVM*, dan *Random Forest*.
   Random Forest menjadi model terbaik.
5. **Tuning (opsional)** — `GridSearchCV` dengan `scoring="precision"`.
6. **Rekomendasi** — content-based (jarak Euclidean pada fitur ter-*scale*),
   disaring oleh prediksi model agar hanya merekomendasikan lagu yang
   diprediksi disukai.

---

## 🛠️ Teknologi

- [Streamlit](https://streamlit.io) — antarmuka web
- [Spotipy](https://spotipy.readthedocs.io) — klien Spotify Web API
- [scikit-learn](https://scikit-learn.org) — machine learning
- [Plotly](https://plotly.com/python/) — visualisasi interaktif
- [pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org)

---

## 📄 Lisensi

Bebas digunakan untuk keperluan pembelajaran dan pengembangan.
