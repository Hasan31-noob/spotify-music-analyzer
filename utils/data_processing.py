"""
data_processing.py
==================
Inti machine learning, dipindahkan & dirapikan dari notebook.

Berisi:
- Pemuatan dataset (data.csv) + fallback dataset sintetis untuk demo.
- Preprocessing sesuai temuan EDA notebook (drop fitur 'energy' karena
  multikolinearitas tinggi dengan 'loudness').
- Training & evaluasi 3 model (Logistic Regression, SVM, Random Forest)
  beserta tuning Random Forest.
- Prediksi "liked" dan rekomendasi berbasis kemiripan audio features.

Semua fungsi berat dibungkus dengan cache Streamlit di app.py agar tidak
dihitung ulang setiap interaksi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Fitur audio yang dipakai (sesuai kolom dataset notebook).
ALL_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "time_signature",
]

# Fitur 'energy' dibuang karena korelasi 0.81 dengan 'loudness' (temuan EDA).
DROPPED_FEATURE = "energy"
MODEL_FEATURES = [f for f in ALL_FEATURES if f != DROPPED_FEATURE]
TARGET = "liked"

# Penjelasan singkat tiap fitur untuk ditampilkan di UI.
FEATURE_DESCRIPTIONS = {
    "danceability": "Seberapa cocok lagu untuk menari (0–1).",
    "energy": "Intensitas & aktivitas lagu (0–1).",
    "key": "Nada dasar lagu (0–11).",
    "loudness": "Kekerasan suara rata-rata dalam desibel (dB).",
    "mode": "Modalitas: 1 = mayor, 0 = minor.",
    "speechiness": "Kehadiran kata/lirik yang diucapkan (0–1).",
    "acousticness": "Tingkat keakustikan lagu (0–1).",
    "instrumentalness": "Probabilitas lagu tanpa vokal (0–1).",
    "liveness": "Indikasi rekaman live / penonton (0–1).",
    "valence": "Tingkat 'kebahagiaan' / positivitas lagu (0–1).",
    "tempo": "Kecepatan lagu dalam BPM.",
    "duration_ms": "Durasi lagu dalam milidetik.",
    "time_signature": "Birama lagu (mis. 3, 4, 5).",
}


# ---------------------------------------------------------------------------
# Pemuatan dataset
# ---------------------------------------------------------------------------
def _generate_demo_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Membuat dataset sintetis yang meniru pola temuan EDA notebook,
    agar aplikasi tetap bisa dijalankan walau data.csv belum tersedia.

    Pola yang ditiru:
    - Lagu disukai  -> danceability, speechiness, valence tinggi.
    - Lagu tak suka -> acousticness, instrumentalness, durasi tinggi.
    """
    rng = np.random.default_rng(seed)
    half = n // 2

    def clip01(x):
        return np.clip(x, 0, 1)

    # Kelas DISUKAI (1)
    liked = pd.DataFrame({
        "danceability": clip01(rng.normal(0.75, 0.12, half)),
        "energy": clip01(rng.normal(0.70, 0.15, half)),
        "key": rng.integers(0, 12, half),
        "loudness": rng.normal(-6.5, 2.5, half),
        "mode": rng.integers(0, 2, half),
        "speechiness": clip01(rng.normal(0.18, 0.10, half)),
        "acousticness": clip01(rng.normal(0.15, 0.12, half)),
        "instrumentalness": clip01(rng.normal(0.02, 0.05, half)),
        "liveness": clip01(rng.normal(0.18, 0.10, half)),
        "valence": clip01(rng.normal(0.60, 0.18, half)),
        "tempo": rng.normal(125, 25, half),
        "duration_ms": rng.normal(200000, 35000, half),
        "time_signature": rng.choice([3, 4, 4, 4, 5], half),
        "liked": 1,
    })

    # Kelas TIDAK DISUKAI (0)
    disliked = pd.DataFrame({
        "danceability": clip01(rng.normal(0.45, 0.15, n - half)),
        "energy": clip01(rng.normal(0.45, 0.22, n - half)),
        "key": rng.integers(0, 12, n - half),
        "loudness": rng.normal(-12.0, 5.0, n - half),
        "mode": rng.integers(0, 2, n - half),
        "speechiness": clip01(rng.normal(0.07, 0.06, n - half)),
        "acousticness": clip01(rng.normal(0.55, 0.28, n - half)),
        "instrumentalness": clip01(rng.normal(0.30, 0.32, n - half)),
        "liveness": clip01(rng.normal(0.22, 0.15, n - half)),
        "valence": clip01(rng.normal(0.42, 0.20, n - half)),
        "tempo": rng.normal(115, 30, n - half),
        "duration_ms": rng.normal(255000, 70000, n - half),
        "time_signature": rng.choice([3, 4, 4, 5], n - half),
        "liked": 0,
    })

    df = pd.concat([liked, disliked], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_dataset(path: str = "data/data.csv") -> tuple[pd.DataFrame, bool]:
    """
    Memuat dataset dari `path`. Jika file tidak ada, mengembalikan dataset
    demo sintetis.

    Returns:
        (df, is_demo) -> is_demo True bila memakai data sintetis.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df, False
    return _generate_demo_dataset(), True


# ---------------------------------------------------------------------------
# Ringkasan dataset (untuk halaman EDA)
# ---------------------------------------------------------------------------
def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Menghitung statistik ringkas untuk ditampilkan sebagai metrik."""
    liked_pct = (df[TARGET].value_counts(normalize=True) * 100).round(2)
    return {
        "n_rows": len(df),
        "n_features": df.shape[1] - 1,
        "n_missing": int(df.isnull().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "pct_liked": float(liked_pct.get(1, 0)),
        "pct_disliked": float(liked_pct.get(0, 0)),
    }


# ---------------------------------------------------------------------------
# Training model
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Wadah hasil training agar mudah dilempar ke UI."""
    scaler: StandardScaler
    best_model: RandomForestClassifier
    feature_names: list[str]
    metrics: dict = field(default_factory=dict)        # nama model -> akurasi
    reports: dict = field(default_factory=dict)        # nama model -> report dict
    conf_matrices: dict = field(default_factory=dict)  # nama model -> cm
    feature_importance: pd.DataFrame = None
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_test: pd.Series = None


def train_models(df: pd.DataFrame, tune: bool = False) -> TrainResult:
    """
    Melatih & mengevaluasi 3 model seperti di notebook.

    Args:
        df: dataset lengkap.
        tune: bila True, lakukan GridSearchCV ringan pada Random Forest.

    Returns:
        TrainResult berisi model terbaik, scaler, metrik, dan importance.
    """
    X = df.drop(columns=[TARGET, DROPPED_FEATURE])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    metrics, reports, conf_matrices = {}, {}, {}
    rf_model = None

    for name, model in models.items():
        # Random Forest tidak butuh scaling (model berbasis tree).
        if name == "Random Forest":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rf_model = model
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        metrics[name] = round(accuracy_score(y_test, y_pred) * 100, 2)
        reports[name] = classification_report(y_test, y_pred, output_dict=True)
        conf_matrices[name] = confusion_matrix(y_test, y_pred)

    # Tuning opsional (dipangkas agar cepat di web).
    best_model = rf_model
    if tune:
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=3, n_jobs=-1, scoring="precision",
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

    # Feature importance dari model terbaik.
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return TrainResult(
        scaler=scaler,
        best_model=best_model,
        feature_names=list(X.columns),
        metrics=metrics,
        reports=reports,
        conf_matrices=conf_matrices,
        feature_importance=importance_df,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
    )


# ---------------------------------------------------------------------------
# Prediksi & rekomendasi
# ---------------------------------------------------------------------------
def predict_like(result: TrainResult, feature_values: dict) -> tuple[int, float]:
    """
    Memprediksi apakah sebuah lagu akan disukai.

    Args:
        result: hasil training.
        feature_values: dict {nama_fitur: nilai} untuk MODEL_FEATURES.

    Returns:
        (label, probabilitas_suka)
    """
    row = pd.DataFrame([[feature_values[f] for f in result.feature_names]],
                       columns=result.feature_names)
    label = int(result.best_model.predict(row)[0])
    proba = float(result.best_model.predict_proba(row)[0][1])
    return label, proba


def recommend_similar(
    df: pd.DataFrame,
    result: TrainResult,
    seed_features: dict,
    n: int = 10,
    only_liked: bool = True,
) -> pd.DataFrame:
    """
    Rekomendasi berbasis konten (content-based) memakai jarak Euclidean
    pada fitur audio yang sudah di-scale.

    Args:
        df: dataset lengkap (sumber kandidat lagu).
        result: hasil training (untuk scaler & model).
        seed_features: profil audio acuan (dict).
        n: jumlah rekomendasi.
        only_liked: bila True, hanya rekomendasikan lagu yang diprediksi disukai.

    Returns:
        DataFrame kandidat terdekat + skor kemiripan.
    """
    candidates = df.copy()
    X = candidates[result.feature_names]
    X_scaled = result.scaler.transform(X)

    seed_row = np.array([[seed_features[f] for f in result.feature_names]])
    seed_scaled = result.scaler.transform(seed_row)

    # Jarak Euclidean -> diubah ke skor kemiripan 0..1.
    distances = np.linalg.norm(X_scaled - seed_scaled, axis=1)
    candidates = candidates.assign(distance=distances)
    candidates["similarity"] = 1 / (1 + candidates["distance"])

    # Saring lagu yang diprediksi disukai oleh model.
    if only_liked:
        preds = result.best_model.predict(X)
        candidates = candidates.assign(pred_liked=preds)
        candidates = candidates[candidates["pred_liked"] == 1]

    return candidates.sort_values("similarity", ascending=False).head(n)
