import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
CLF_PATH    = os.path.join(BASE, "id3_model.pkl")
LABELS_PATH = os.path.join(BASE, "class_labels.pkl")
CLUST_PATH  = os.path.join(BASE, "cluster_model.pkl")
SCALER_PATH = os.path.join(BASE, "scaler.pkl")
FEATS_PATH  = os.path.join(BASE, "features.pkl")
CSTATS_PATH = os.path.join(BASE, "cluster_stats.pkl")

# ── Genre list (114 genres from dataset) ──────────────────────────────────────
GENRES = [
    "acoustic","afrobeat","alt-rock","alternative","ambient","anime",
    "black-metal","bluegrass","blues","brazil","breakbeat","british",
    "cantopop","chicago-house","children","chill","classical","club",
    "comedy","country","dance","dancehall","death-metal","deep-house",
    "detroit-techno","disco","disney","drum-and-bass","dub","dubstep",
    "edm","electro","electronic","emo","folk","forro","french","funk",
    "garage","german","gospel","goth","grindcore","groove","grunge",
    "guitar","happy","hard-rock","hardcore","hardstyle","heavy-metal",
    "hip-hop","honky-tonk","house","idm","indian","indie","indie-pop",
    "industrial","iranian","j-dance","j-idol","j-pop","j-rock","jazz",
    "k-pop","kids","latin","latino","malay","mandopop","metal","metalcore",
    "minimal-techno","mpb","new-age","opera","pagode","party","piano",
    "pop","pop-film","power-pop","progressive-house","psych-rock","punk",
    "punk-rock","r-n-b","reggae","reggaeton","rock","rock-n-roll",
    "rockabilly","romance","sad","salsa","samba","sertanejo","show-tunes",
    "singer-songwriter","ska","sleep","songwriter","soul","spanish","study",
    "swedish","synth-pop","tango","techno","trance","trip-hop","turkish",
    "world-music",
]

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #f8fafc; min-height: 100vh; }
  #MainMenu, footer, header { visibility: hidden; }

  .card {
      background: #ffffff;
      border-radius: 20px;
      padding: 2rem 2.5rem;
      box-shadow: 0 4px 30px rgba(100,80,200,.08);
      border: 1px solid #e8e0ff;
      margin-bottom: 1.5rem;
  }
  .hero-title {
      font-size: 2.6rem;
      font-weight: 800;
      background: linear-gradient(90deg, #7c3aed, #ec4899, #f97316);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-align: center;
      margin-bottom: 0.3rem;
  }
  .hero-sub { text-align: center; color: #000000; font-size: 1.05rem; margin-bottom: 2rem; }
  .section-title { font-size: 1.35rem; font-weight: 700; color: #4c1d95; margin-bottom: 1rem; }

  .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin: 1.5rem 0; }
  .mode-card {
      background: linear-gradient(135deg, #ede9fe, #fce7f3);
      border-radius: 16px; padding: 1.5rem; text-align: center;
      border: 2px solid transparent; cursor: pointer; transition: all .2s;
  }
  .mode-card:hover { border-color: #7c3aed; box-shadow: 0 6px 20px rgba(124,58,237,.15); transform: translateY(-2px); }
  .mode-icon { font-size: 2.5rem; margin-bottom: .5rem; }
  .mode-label { font-weight: 700; color: #4c1d95; font-size: 1.1rem; }
  .mode-desc  { color: #1f2937; font-size: .85rem; margin-top: .3rem; }

  .result-badge {
      background: linear-gradient(135deg, #7c3aed, #ec4899);
      color: white; border-radius: 50px; padding: 0.9rem 2rem;
      font-size: 1.4rem; font-weight: 800; text-align: center;
      margin: 1.5rem auto; display: block;
      box-shadow: 0 4px 20px rgba(124,58,237,.3);
  }
  .result-sub { text-align: center; color: #1f2937; font-size: .95rem; }

  /* All input labels */
  .stSlider label, .stSlider p,
  .stSelectbox label, .stSelectbox p,
  .stRadio label, .stRadio p,
  .stNumberInput label, .stNumberInput p,
  div[data-testid="stWidgetLabel"] p,
  div[data-testid="stWidgetLabel"] label { color: #111827 !important; font-weight: 600 !important; }
  .stRadio div[role="radiogroup"] label span { color: #111827 !important; font-weight: 500 !important; }
  div[data-baseweb="select"] span { color: #111827 !important; }

  /* Metric (track summary) */
  div[data-testid="stMetricLabel"] p,
  div[data-testid="stMetricLabel"] { color: #111827 !important; font-weight: 700 !important; font-size: 0.9rem !important; }
  div[data-testid="stMetricValue"] { color: #4c1d95 !important; font-weight: 800 !important; }

  div[data-testid="stButton"] > button {
      background: linear-gradient(90deg, #7c3aed, #ec4899) !important;
      color: white !important; border: none !important;
      border-radius: 50px !important; padding: .6rem 2.5rem !important;
      font-size: 1.05rem !important; font-weight: 700 !important;
      letter-spacing: .03em;
      box-shadow: 0 4px 15px rgba(124,58,237,.25) !important;
      transition: all .2s !important; width: 100%;
  }
  div[data-testid="stButton"] > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(124,58,237,.35) !important;
  }
  .back-btn > button {
      background: transparent !important; color: #7c3aed !important;
      border: 2px solid #d8b4fe !important; box-shadow: none !important;
      padding: .35rem 1.2rem !important; font-size: .9rem !important;
      font-weight: 600 !important; width: auto !important;
  }
  .pop-bar-wrap { background: #ede9fe; border-radius: 50px; height: 14px; margin-top: .5rem; }
  .pop-bar-fill { background: linear-gradient(90deg, #7c3aed, #ec4899); border-radius: 50px; height: 14px; transition: width .6s ease; }
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf    = joblib.load(CLF_PATH)
    labels = joblib.load(LABELS_PATH)
    clust  = joblib.load(CLUST_PATH)
    scaler = joblib.load(SCALER_PATH)
    feats  = joblib.load(FEATS_PATH)
    cstats = joblib.load(CSTATS_PATH)
    return clf, labels, clust, scaler, feats, cstats

clf, class_labels, clust_model, scaler, features, cluster_stats = load_models()

# ── Session state ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

# ── Helper: ms → "Xm Ys" ──────────────────────────────────────────────────────
def ms_to_label(ms):
    total_s = int(ms) // 1000
    return f"{total_s // 60}m {total_s % 60}s"

# ── Shared input form ──────────────────────────────────────────────────────────
def render_inputs():
    """
    Collects user input in human-readable units.
    Returns (vals_dict, genre_str).
    vals_dict has values ready for model inference:
      - duration_ms in milliseconds (converted from minutes)
      - 0-1 features as floats (sliders shown as 0-100 %)
      - explicit as 0/1 int
    genre is UI-only (not fed to the model, which was trained without it).
    """
    vals = {}

    # ── Track Info row ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎵 Track Info</div>', unsafe_allow_html=True)
    col_genre, col_dur = st.columns(2)
    with col_genre:
        genre = st.selectbox(
            "Genre",
            GENRES,
            index=GENRES.index("pop"),
            help="Track genre (used for display — 114 genres from the dataset)"
        )
    with col_dur:
        dur_min = st.number_input(
            "Duration (minutes)",
            min_value=0.1, max_value=90.0,
            value=3.5, step=0.1, format="%.1f",
            help="Enter track length in minutes (e.g. 3.5 = 3 min 30 sec)"
        )
    vals['duration_ms'] = dur_min * 60 * 1000   # convert to ms for model

    # ── Audio Features ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.2rem;">🎚️ Audio Features</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        vals['danceability']     = st.slider("Danceability (%)",     0, 100, 60, 1,
                                    help="How suitable for dancing (0 = not danceable, 100 = very danceable)") / 100
        vals['energy']           = st.slider("Energy (%)",           0, 100, 70, 1,
                                    help="Perceptual intensity and activity") / 100
        vals['speechiness']      = st.slider("Speechiness (%)",      0, 100,  5, 1,
                                    help="Presence of spoken words (>66% = mostly speech)") / 100
        vals['acousticness']     = st.slider("Acousticness (%)",     0, 100, 20, 1,
                                    help="Confidence that the track is acoustic") / 100
        vals['instrumentalness'] = st.slider("Instrumentalness (%)", 0, 100,  0, 1,
                                    help="Likelihood of no vocals (>50% = likely instrumental)") / 100
        vals['liveness']         = st.slider("Liveness (%)",         0, 100, 15, 1,
                                    help="Detects presence of a live audience (>80% = likely live)") / 100
        vals['valence']          = st.slider("Valence / Mood (%)",   0, 100, 50, 1,
                                    help="Musical positiveness (0 = sad/angry, 100 = happy/euphoric)") / 100

    with c2:
        vals['loudness']       = st.slider("Loudness (dB)", -50.0, 5.0, -8.0, 0.1,
                                    help="Overall loudness in dB (typical range: -60 to 0 dB)")
        vals['tempo']          = st.slider("Tempo (BPM)",   0.0, 250.0, 120.0, 0.5,
                                    help="Estimated tempo in beats per minute")
        vals['key']            = st.selectbox("Key", list(range(12)),
                                    format_func=lambda x: ['C','C#','D','D#','E','F',
                                                            'F#','G','G#','A','A#','B'][x],
                                    help="Musical key of the track")
        vals['mode']           = st.radio("Mode", [0, 1],
                                    format_func=lambda x: "Minor" if x == 0 else "Major",
                                    horizontal=True)
        vals['time_signature'] = st.selectbox("Time Signature", [3, 4, 5],
                                    format_func=lambda x: f"{x}/4",
                                    help="Estimated beats per measure")
        vals['explicit']       = st.radio("Explicit?", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes",
                                    horizontal=True)

    return vals, genre


def vals_to_df(vals):
    return pd.DataFrame([[vals[f] for f in features]], columns=features)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown('<div class="hero-title">🎵 Spotify Popularity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Explore your track\'s predicted popularity using ML</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div style="text-align:center; color:#4c1d95; font-weight:700; font-size:1.1rem; margin-bottom:1rem;">
            Choose a prediction mode to get started
        </div>
        <div class="mode-grid">
            <div class="mode-card">
                <div class="mode-icon">🌿</div>
                <div class="mode-label">Classification</div>
                <div class="mode-desc">ID3 Decision Tree — predicts a popularity tier (Very Low → Very High)</div>
            </div>
            <div class="mode-card">
                <div class="mode-icon">🔵</div>
                <div class="mode-label">Clustering</div>
                <div class="mode-desc">K-Means++ — groups your track with similar songs and shows expected popularity</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌿  Go to Classification"):
            st.session_state.page = "classification"
            st.rerun()
    with col2:
        if st.button("🔵  Go to Clustering"):
            st.session_state.page = "clustering"
            st.rerun()

    st.markdown("""
    <div class="card" style="margin-top:1.5rem;">
        <div style="font-weight:700; color:#4c1d95; margin-bottom:.6rem;">📊 Dataset Overview</div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; text-align:center;">
            <div><div style="font-size:1.6rem; font-weight:800; color:#7c3aed;">114 K</div>
                 <div style="color:#1f2937;font-size:.85rem;font-weight:600;">Tracks</div></div>
            <div><div style="font-size:1.6rem; font-weight:800; color:#ec4899;">114</div>
                 <div style="color:#1f2937;font-size:.85rem;font-weight:600;">Genres</div></div>
            <div><div style="font-size:1.6rem; font-weight:800; color:#f97316;">5</div>
                 <div style="color:#1f2937;font-size:.85rem;font-weight:600;">Popularity Tiers</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def page_classification():
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-title" style="font-size:2rem;">🌿 Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ID3 Decision Tree predicts your popularity tier</div>', unsafe_allow_html=True)

    tier_colours = {
        "Very Low":  ("#fef2f2", "#dc2626"),
        "Low":       ("#fff7ed", "#ea580c"),
        "Medium":    ("#fefce8", "#ca8a04"),
        "High":      ("#f0fdf4", "#16a34a"),
        "Very High": ("#f5f3ff", "#7c3aed"),
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    vals, genre = render_inputs()
    predict_clicked = st.button("🎯  Predict Popularity Tier")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        X_in    = vals_to_df(vals)
        pred    = clf.predict(X_in)[0]
        proba   = clf.predict_proba(X_in)[0]
        classes = clf.classes_
        _, fg   = tier_colours.get(pred, ("#f5f3ff", "#7c3aed"))
        dur_label = ms_to_label(vals['duration_ms'])

        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="font-size:.9rem; color:#1f2937; font-weight:600; margin-bottom:.5rem;">
                Predicted Popularity Tier &nbsp;·&nbsp;
                <span style="color:#7c3aed;">{genre}</span>
                &nbsp;·&nbsp; <span style="color:#6b21a8;">{dur_label}</span>
            </div>
            <div class="result-badge">{pred}</div>
            <div class="result-sub">Confidence: {max(proba)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Class Probabilities</div>', unsafe_allow_html=True)
        for cls, p in zip(classes, proba):
            _, fg2 = tier_colours.get(cls, ("#f5f3ff", "#7c3aed"))
            st.markdown(f"""
            <div style="margin-bottom:.8rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:.2rem;">
                    <span style="font-weight:700; color:{fg2};">{cls}</span>
                    <span style="color:#1f2937; font-size:.9rem; font-weight:600;">{p*100:.1f}%</span>
                </div>
                <div class="pop-bar-wrap">
                    <div class="pop-bar-fill" style="width:{p*100:.1f}%; background:{fg2};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature importances
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔑 Top Feature Importances</div>', unsafe_allow_html=True)
        importances = clf.feature_importances_
        fi = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:5]
        for fname, fval in fi:
            st.markdown(f"""
            <div style="margin-bottom:.8rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:.2rem;">
                    <span style="font-weight:700; color:#4c1d95;">{fname}</span>
                    <span style="color:#1f2937; font-size:.9rem; font-weight:600;">{fval*100:.1f}%</span>
                </div>
                <div class="pop-bar-wrap">
                    <div class="pop-bar-fill" style="width:{fval*100:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
def page_clustering():
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-title" style="font-size:2rem;">🔵 Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">K-Means++ groups your track with similar songs</div>', unsafe_allow_html=True)

    cluster_names = {
        0: ("🎸 Energetic",        "#7c3aed"),
        1: ("🎹 Melodic",          "#ec4899"),
        2: ("🌿 Chill / Acoustic", "#16a34a"),
        3: ("🎤 Vocal-driven",     "#ea580c"),
        4: ("🔥 Mainstream",       "#f97316"),
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    vals, genre = render_inputs()
    predict_clicked = st.button("🔍  Find My Cluster")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        X_in       = vals_to_df(vals)
        X_sc       = scaler.transform(X_in)
        cluster_id = int(clust_model.predict(X_sc)[0])
        cname, ccolor = cluster_names.get(cluster_id, (f"Cluster {cluster_id}", "#7c3aed"))
        avg_pop    = cluster_stats.get(cluster_id, 0)
        dur_label  = ms_to_label(vals['duration_ms'])

        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="font-size:.9rem; color:#1f2937; font-weight:600; margin-bottom:.5rem;">
                Your track belongs to &nbsp;·&nbsp;
                <span style="color:#7c3aed;">{genre}</span>
                &nbsp;·&nbsp; <span style="color:#6b21a8;">{dur_label}</span>
            </div>
            <div class="result-badge" style="background:{ccolor};">{cname}</div>
            <div class="result-sub">Cluster #{cluster_id}</div>
        </div>
        """, unsafe_allow_html=True)

        # Expected popularity bar
        st.markdown(f"""
        <div class="card">
            <div class="section-title">📈 Expected Popularity Score</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:.4rem;">
                <span style="font-weight:700; font-size:1.6rem; color:{ccolor};">{avg_pop:.1f}
                    <span style="font-size:1rem; color:#1f2937; font-weight:600;"> / 100</span>
                </span>
                <span style="color:#1f2937; font-size:.9rem; font-weight:600; align-self:flex-end;">Cluster average</span>
            </div>
            <div class="pop-bar-wrap">
                <div class="pop-bar-fill" style="width:{avg_pop}%; background:{ccolor};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Cluster comparison
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗺️ Cluster Comparison</div>', unsafe_allow_html=True)
        for cid, (cname2, ccolor2) in cluster_names.items():
            pop2      = cluster_stats.get(cid, 0)
            highlight = f"border: 2px solid {ccolor2};" if cid == cluster_id else ""
            st.markdown(f"""
            <div style="background:#f9f5ff; border-radius:12px; padding:.8rem 1rem; margin-bottom:.6rem; {highlight}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.3rem;">
                    <span style="font-weight:700; color:{ccolor2};">{cname2}</span>
                    <span style="color:#1f2937; font-size:.85rem; font-weight:600;">avg pop: {pop2:.1f}</span>
                </div>
                <div class="pop-bar-wrap" style="height:10px;">
                    <div class="pop-bar-fill" style="width:{pop2}%; height:10px; background:{ccolor2};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Track summary in real units (2 rows of 5)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Your Track Summary</div>', unsafe_allow_html=True)

        row1 = st.columns(5)
        for col, (label, value) in zip(row1, [
            ("Genre",        genre),
            ("Duration",     dur_label),
            ("Danceability", f"{vals['danceability']*100:.0f}%"),
            ("Energy",       f"{vals['energy']*100:.0f}%"),
            ("Valence",      f"{vals['valence']*100:.0f}%"),
        ]):
            col.metric(label, value)

        row2 = st.columns(5)
        for col, (label, value) in zip(row2, [
            ("Acousticness",     f"{vals['acousticness']*100:.0f}%"),
            ("Tempo",            f"{vals['tempo']:.0f} BPM"),
            ("Loudness",         f"{vals['loudness']:.1f} dB"),
            ("Speechiness",      f"{vals['speechiness']*100:.0f}%"),
            ("Instrumentalness", f"{vals['instrumentalness']*100:.0f}%"),
        ]):
            col.metric(label, value)

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "home":
    page_home()
elif page == "classification":
    page_classification()
elif page == "clustering":
    page_clustering()
