"""
app.py
======
Aplikasi web Streamlit: Spotify Music Analyzer & Recommender.

Versi ini fokus pada analisis dataset & machine learning (tanpa Spotify API):
- Analisis data (EDA), perbandingan model, feature importance.
- Prediksi "liked" dan rekomendasi lagu berbasis audio features.

Cara menjalankan:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import data_processing as dp
from utils import visualization as viz

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Music Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS kustom untuk tampilan modern bertema Spotify.
st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #121212 0%, #1a1a1a 100%); }
        h1, h2, h3 { color: #FFFFFF; font-weight: 700; }
        .accent { color: #1DB954; }
        .hero {
            background: linear-gradient(135deg, #1DB954 0%, #0d7a36 100%);
            padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1.5rem;
            box-shadow: 0 8px 30px rgba(29,185,84,0.25);
        }
        .hero h1 { color: #fff; margin: 0; font-size: 2.4rem; }
        .hero p { color: #eafff0; margin: 0.5rem 0 0; font-size: 1.05rem; }
        .card {
            background: #181818; border-radius: 14px; padding: 1rem;
            border: 1px solid #282828; transition: 0.2s; height: 100%;
        }
        .card:hover { background: #222; border-color: #1DB954; }
        .stButton>button {
            background: #1DB954; color: #fff; border: none; border-radius: 24px;
            padding: 0.5rem 1.4rem; font-weight: 600;
        }
        .stButton>button:hover { background: #1ed760; color: #000; }
        div[data-testid="stMetricValue"] { color: #1DB954; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cache: dataset & model dilatih sekali saja
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_data():
    return dp.load_dataset()


@st.cache_resource(show_spinner="Melatih model machine learning...")
def _train(_df: pd.DataFrame, tune: bool):
    return dp.train_models(_df, tune=tune)


# ---------------------------------------------------------------------------
# Sidebar — navigasi & pengaturan
# ---------------------------------------------------------------------------
with st.sidebar:
    import os
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=160)
    st.markdown("## 🎵 Music Analyzer")

    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "📊 Analisis Data", "🤖 Prediksi & Rekomendasi", "ℹ️ Tentang"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan")
    use_tuning = st.checkbox("Aktifkan tuning model (lebih lambat)", value=False)


# Muat data sekali.
df, is_demo = _load_data()


# ---------------------------------------------------------------------------
# Halaman: Beranda
# ---------------------------------------------------------------------------
def page_home():
    st.markdown(
        """
        <div class="hero">
            <h1>🎵 Spotify Music Analyzer & Recommender</h1>
            <p>Analisis selera musik dengan machine learning dan dapatkan
            rekomendasi lagu berbasis audio features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_demo:
        st.info(
            "📦 Saat ini aplikasi memakai **dataset demo sintetis** karena "
            "`data/data.csv` belum ditemukan. Letakkan dataset asli Anda di "
            "`data/data.csv` untuk hasil analisis yang sesungguhnya.",
            icon="ℹ️",
        )

    overview = dp.get_dataset_overview(df)
    st.markdown("### Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Lagu", f"{overview['n_rows']:,}")
    c2.metric("Jumlah Fitur", overview["n_features"])
    c3.metric("Disukai", f"{overview['pct_liked']:.1f}%")
    c4.metric("Tidak Disukai", f"{overview['pct_disliked']:.1f}%")

    st.markdown("### Apa yang bisa Anda lakukan di sini?")
    a, b = st.columns(2)
    with a:
        st.markdown(
            "<div class='card'><h3>📊 Analisis</h3>"
            "<p>Eksplorasi selera musik lewat radar chart, korelasi, PCA, "
            "dan perbandingan 3 model machine learning.</p></div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            "<div class='card'><h3>🤖 Rekomendasi</h3>"
            "<p>Atur profil audio lagu favorit Anda, lalu biarkan model "
            "merekomendasikan lagu serupa yang kemungkinan Anda sukai.</p></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Halaman: Analisis Data (EDA + model)
# ---------------------------------------------------------------------------
def page_analysis():
    st.markdown("## 📊 Analisis Data & Model")
    st.caption("Visualisasi & model machine learning dari dataset audio features.")

    result = _train(df, use_tuning)

    tab1, tab2, tab3 = st.tabs(
        ["Eksplorasi (EDA)", "Korelasi & PCA", "Model & Evaluasi"]
    )

    with tab1:
        st.plotly_chart(viz.plot_target_balance(df), width='stretch')
        st.markdown(
            "> Dataset relatif **seimbang**, sehingga akurasi layak dipakai "
            "sebagai metrik utama tanpa perlu oversampling."
        )
        st.plotly_chart(viz.plot_kde_features(df), width='stretch')
        st.plotly_chart(viz.plot_radar(df), width='stretch')

    with tab2:
        st.plotly_chart(viz.plot_correlation(df), width='stretch')
        st.markdown(
            "> Terdapat korelasi tinggi antara **energy** & **loudness**. "
            "Fitur `energy` dibuang sebelum training untuk menghindari "
            "multikolinearitas."
        )
        st.plotly_chart(viz.plot_pca(df), width='stretch')

    with tab3:
        st.plotly_chart(viz.plot_model_comparison(result.metrics), width='stretch')
        best_name = max(result.metrics, key=result.metrics.get)
        st.success(f"Model terbaik: **{best_name}** "
                   f"({result.metrics[best_name]}% akurasi)")

        st.plotly_chart(viz.plot_feature_importance(result.feature_importance),
                        width='stretch')

        st.markdown("### Confusion Matrix per Model")
        cm_cols = st.columns(3)
        for i, (name, cm) in enumerate(result.conf_matrices.items()):
            with cm_cols[i]:
                st.plotly_chart(viz.plot_confusion_matrix(cm, name),
                                width='stretch')


# ---------------------------------------------------------------------------
# Halaman: Prediksi & Rekomendasi
# ---------------------------------------------------------------------------
def page_recommend():
    st.markdown("## 🤖 Prediksi & Rekomendasi")
    result = _train(df, use_tuning)

    st.markdown("### 1. Atur profil audio lagu")
    st.caption("Geser slider untuk mendeskripsikan lagu, lalu prediksi & "
               "dapatkan rekomendasi serupa.")

    # Nilai awal slider diambil dari rata-rata dataset.
    defaults = df[result.feature_names].mean()
    values: dict = {}

    col1, col2 = st.columns(2)
    half = len(result.feature_names) // 2 + 1
    for i, feat in enumerate(result.feature_names):
        target_col = col1 if i < half else col2
        with target_col:
            series = df[feat]
            fmin, fmax = float(series.min()), float(series.max())
            step = 0.01 if fmax <= 1.0 else (1.0 if fmax > 100 else 0.1)
            values[feat] = st.slider(
                f"{feat}",
                min_value=round(fmin, 2),
                max_value=round(fmax, 2),
                value=round(float(defaults[feat]), 2),
                step=step,
                help=dp.FEATURE_DESCRIPTIONS.get(feat, ""),
            )

    if st.button("🎯 Prediksi & Rekomendasikan", type="primary"):
        label, proba = dp.predict_like(result, values)

        st.markdown("### 2. Hasil Prediksi")
        if label == 1:
            st.success(f"✅ Lagu ini kemungkinan **DISUKAI** "
                       f"(probabilitas {proba*100:.1f}%).")
        else:
            st.error(f"❌ Lagu ini kemungkinan **TIDAK DISUKAI** "
                     f"(probabilitas suka {proba*100:.1f}%).")
        st.progress(proba)

        st.markdown("### 3. Rekomendasi Lagu Serupa")
        recs = dp.recommend_similar(df, result, values, n=10, only_liked=True)
        if recs.empty:
            st.warning("Tidak ada lagu serupa yang diprediksi disukai. "
                       "Coba ubah profil audio.")
        else:
            show = recs[result.feature_names + ["similarity"]].copy()
            show["similarity"] = (show["similarity"] * 100).round(1)
            show = show.rename(columns={"similarity": "Kemiripan (%)"})
            st.dataframe(show.reset_index(drop=True), width='stretch')


# ---------------------------------------------------------------------------
# Halaman: Tentang
# ---------------------------------------------------------------------------
def page_about():
    st.markdown("## ℹ️ Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini dikembangkan dari notebook analisis *Spotify Music
        Recommendation*. Tujuannya memprediksi apakah seorang pengguna akan
        **menyukai** sebuah lagu berdasarkan **audio features**-nya, lalu
        memberi rekomendasi lagu serupa.

        **Alur machine learning:**
        - **EDA** — cek balance target, distribusi fitur, outlier sebagai sinyal.
        - **Feature selection** — membuang `energy` (korelasi tinggi dengan
          `loudness`).
        - **Preprocessing** — `train_test_split` 80:20 + `StandardScaler`.
        - **Model** — membandingkan *Logistic Regression*, *SVM*, dan
          *Random Forest*. Random Forest menjadi model terbaik.
        - **Rekomendasi** — content-based (jarak Euclidean pada fitur
          ter-*scale*), disaring oleh prediksi model.

        **Teknologi:** Streamlit · scikit-learn · Plotly · pandas.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Beranda": page_home,
    "📊 Analisis Data": page_analysis,
    "🤖 Prediksi & Rekomendasi": page_recommend,
    "ℹ️ Tentang": page_about,
}

PAGES[page]()