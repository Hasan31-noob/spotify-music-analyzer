"""
visualization.py
================
Semua visualisasi dipindahkan dari notebook (yang memakai matplotlib/seaborn)
ke Plotly agar interaktif dan tampil modern di web.

Setiap fungsi mengembalikan objek figure Plotly yang siap dirender dengan
st.plotly_chart(...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Palet warna konsisten (hijau = suka, merah = tidak suka).
COLOR_LIKED = "#1DB954"      # hijau Spotify
COLOR_DISLIKED = "#E0245E"   # merah
TEMPLATE = "plotly_dark"


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Menerapkan gaya dasar yang seragam ke semua figure."""
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color="#E8E8E8"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_target_balance(df: pd.DataFrame) -> go.Figure:
    """Bar chart proporsi target (liked vs disliked)."""
    counts = df["liked"].value_counts().sort_index()
    labels = ["Tidak Suka (0)", "Suka (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[COLOR_DISLIKED, COLOR_LIKED],
            text=values,
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Jumlah Lagu")
    return _base_layout(fig, "Proporsi Target Variable")


def plot_kde_features(df: pd.DataFrame, features: list[str] | None = None) -> go.Figure:
    """
    Distribusi (histogram bertumpuk) beberapa fitur audio dipisah
    berdasarkan label, meniru KDE plot di notebook.
    """
    if features is None:
        features = ["danceability", "energy", "valence",
                    "acousticness", "speechiness", "tempo"]

    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols

    from plotly.subplots import make_subplots

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f.capitalize() for f in features])

    for i, feat in enumerate(features):
        r, c = i // n_cols + 1, i % n_cols + 1
        for label, color, name in [(0, COLOR_DISLIKED, "Tidak Suka"),
                                    (1, COLOR_LIKED, "Suka")]:
            subset = df[df["liked"] == label][feat]
            fig.add_trace(
                go.Histogram(
                    x=subset, name=name, marker_color=color, opacity=0.55,
                    legendgroup=name, showlegend=(i == 0), nbinsx=30,
                ),
                row=r, col=c,
            )

    fig.update_layout(barmode="overlay", height=300 * n_rows)
    return _base_layout(fig, "Distribusi Fitur Audio (Suka vs Tidak Suka)")


def plot_radar(df: pd.DataFrame) -> go.Figure:
    """Radar chart profil rata-rata fitur (skala 0–1)."""
    radar_features = ["danceability", "energy", "valence",
                      "acousticness", "speechiness", "liveness"]
    means = df.groupby("liked")[radar_features].mean()
    categories = [f.capitalize() for f in radar_features]

    fig = go.Figure()
    for label, color, name in [(0, COLOR_DISLIKED, "Tidak Suka"),
                               (1, COLOR_LIKED, "Suka")]:
        if label in means.index:
            values = means.loc[label].tolist()
            fig.add_trace(go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself", name=name,
                line_color=color,
            ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    return _base_layout(fig, "Profil Rata-rata Lagu (Radar Chart)")


def plot_correlation(df: pd.DataFrame) -> go.Figure:
    """Heatmap matriks korelasi antar fitur numerik."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().round(2)

    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    return _base_layout(fig, "Matriks Korelasi Fitur Audio")


def plot_pca(df: pd.DataFrame) -> go.Figure:
    """Proyeksi PCA 2D dari seluruh fitur audio."""
    X = df.drop(columns=["liked"])
    y = df["liked"]

    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Label": y.map({0: "Tidak Suka", 1: "Suka"}),
    })

    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Label",
        color_discrete_map={"Suka": COLOR_LIKED, "Tidak Suka": COLOR_DISLIKED},
        opacity=0.6,
    )
    return _base_layout(fig, "Proyeksi PCA 2D dari Fitur Audio")


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Bar chart horizontal tingkat kepentingan fitur (Random Forest)."""
    data = importance_df.sort_values("Importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["Importance"], y=data["Feature"],
            orientation="h", marker_color=COLOR_LIKED,
            text=data["Importance"].round(3), textposition="outside",
        )
    )
    fig.update_xaxes(title="Importance Score")
    return _base_layout(fig, "Feature Importance — Random Forest")


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> go.Figure:
    """Heatmap confusion matrix untuk satu model."""
    labels = ["Tidak Suka (0)", "Suka (1)"]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Greens",
        x=labels, y=labels, aspect="auto",
        labels=dict(x="Prediksi", y="Label Asli", color="Jumlah"),
    )
    return _base_layout(fig, f"Confusion Matrix — {model_name}")


def plot_model_comparison(metrics: dict) -> go.Figure:
    """Bar chart perbandingan akurasi antar model."""
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = [COLOR_LIKED if v == max(values) else "#5A5A5A" for v in values]

    fig = go.Figure(
        go.Bar(
            x=names, y=values, marker_color=colors,
            text=[f"{v}%" for v in values], textposition="outside",
        )
    )
    fig.update_yaxes(title="Akurasi (%)", range=[0, 100])
    return _base_layout(fig, "Perbandingan Akurasi Model")


def plot_popularity_bar(tracks: list[dict]) -> go.Figure:
    """Bar chart popularity hasil pencarian Spotify (live data)."""
    df = pd.DataFrame(tracks)
    df = df.sort_values("popularity", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["popularity"], y=df["name"],
            orientation="h", marker_color=COLOR_LIKED,
            text=df["popularity"], textposition="outside",
            hovertext=df["artist"],
        )
    )
    fig.update_xaxes(title="Popularity Score (0–100)", range=[0, 105])
    fig.update_layout(height=max(300, 45 * len(df)))
    return _base_layout(fig, "Popularity Lagu Hasil Pencarian")
