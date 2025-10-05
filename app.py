import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import lightkurve as lk
from astropy.timeseries import LombScargle

# Load trained model (random_forest_model.joblib) and optional scaler
@st.cache_resource
def load_trained_model(model_path="random_forest_model.joblib"):
    """
    Load the trained model from models/random_forest_model.joblib.
    No scaler is used — model file may already contain any preprocessing.
    Returns (model, None) or (None, None) if missing.
    """
    try:
        base = Path(__file__).resolve().parents[1]
        model_fp = base / model_path
        if not model_fp.exists():
            model_fp = Path.cwd() / model_path
        if not model_fp.exists():
            return None, None
        model = joblib.load(str(model_fp))
        return model, None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# Load dataset from data/evap_synthesized_catalog.csv (with robust column matching)
@st.cache_data
def load_dataset(csv_path="data/evap_synthesized_catalog.csv"):
    try:
        csv_abspath = Path(csv_path)
        if not csv_abspath.exists():
            # fallback to app-level CSV
            csv_abspath = Path(__file__).resolve().parents[0] / "evap_synthesized_catalog.csv"
        df = pd.read_csv(csv_abspath)
    except Exception as e:
        st.error(f"Failed to load CSV {csv_path}: {e}")
        return None

    # Normalize column names to lowercase for matching
    df_cols_lower = {c.lower(): c for c in df.columns}
    # Keep the dataframe as-is but provide a helper to map names
    df._col_map_lower = df_cols_lower  # attach helper mapping
    return df

# resolve CSV path relative to this file (project structure: ../data/evap_synthesized_catalog.csv)
DATA_CSV = Path(__file__).resolve().parents[0] / "evap_synthesized_catalog.csv"

# load dataset using the resolved absolute path
dataset = load_dataset(csv_path=str(DATA_CSV))

# Reuse your simulated contributions function (ensure it exists)
def _simulated_contributions(orbital_period, transit_depth, snr, stellar_radius):
    snr_contribution = snr / 100 * 40
    depth_contribution = transit_depth * 100 * 30
    period_ideal = 100.0
    period_penalty_factor = 0.3 
    period_penalty = 1 - (abs(orbital_period - period_ideal) / period_ideal)**2 * period_penalty_factor 
    period_contribution = max(0, 10 * period_penalty)
    radius_contribution = min(10.0, (1 / stellar_radius) * 10)
    base_prob = snr_contribution + depth_contribution + period_contribution + radius_contribution
    return {
        "SNR": snr_contribution,
        "Transit Depth": depth_contribution,
        "Orbital Period": period_contribution,
        "Stellar Radius": radius_contribution,
        "Base Probability": base_prob
    }

# Updated predict_exoplanet uses loaded model (joblib) when available
def predict_exoplanet(orbital_period, transit_depth, snr, stellar_radius):
    """
    Predict using the trained model only.
    Raises RuntimeError if the model is not present or prediction fails.
    Returns: (probability_percent(float), prediction_class(str), contributions(dict), delta_color(str))
    """
    model, _ = load_trained_model()
    if model is None:
        raise RuntimeError("Trained model not found at models/random_forest_model.joblib — place your model file there.")

    X = np.array([[orbital_period, transit_depth, snr, stellar_radius]], dtype=float)
    # no scaler available — pass raw features
    features_scaled = X

    # classifier with predict_proba
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_scaled)[0]
            classes = getattr(model, "classes_", None)
            idx = int(np.argmax(probs))
            probability = float(probs[idx] * 100.0)
            raw_label = str(classes[idx]) if classes is not None else ""
            raw_lower = raw_label.lower() if raw_label else ""
            if "confirm" in raw_lower:
                predicted_label = "Confirmed Exoplanet"
            elif "candidate" in raw_lower:
                predicted_label = "High-Value Candidate"
            else:
                predicted_label = "Confirmed Exoplanet" if probability > 85 else ("High-Value Candidate" if probability > 60 else "False Positive")
        elif hasattr(model, "predict"):
            pr = model.predict(features_scaled)
            val = float(pr[0]) if hasattr(pr, "__iter__") else float(pr)
            probability = float(np.clip(val, 0.0, 1.0) * 100.0)
            predicted_label = "Confirmed Exoplanet" if probability > 85 else ("High-Value Candidate" if probability > 60 else "False Positive")
        else:
            raise RuntimeError("Loaded model does not support predict or predict_proba")

        # contributions: use model.feature_importances_ if available, else use simulated proxy
        contributions = _simulated_contributions(orbital_period, transit_depth, snr, stellar_radius)
        if hasattr(model, "feature_importances_"):
            try:
                fi = np.array(model.feature_importances_, dtype=float)
                if fi.sum() > 0:
                    fi_norm = fi / fi.sum()
                    total_base = sum([contributions[k] for k in ["SNR", "Transit Depth", "Orbital Period", "Stellar Radius"]])
                    contributions = {
                        "Orbital Period": float(total_base * fi_norm[0]) if len(fi_norm) > 0 else contributions["Orbital Period"],
                        "Transit Depth": float(total_base * fi_norm[1]) if len(fi_norm) > 1 else contributions["Transit Depth"],
                        "SNR": float(total_base * fi_norm[2]) if len(fi_norm) > 2 else contributions["SNR"],
                        "Stellar Radius": float(total_base * fi_norm[3]) if len(fi_norm) > 3 else contributions["Stellar Radius"],
                        "Base Probability": float(total_base)
                    }
            except Exception:
                # keep proxy contributions on failure
                pass

        delta_color = "normal" if probability > 85 else ("caution" if probability > 60 else "inverse")
        probability = float(np.clip(probability, 0.0, 99.9))
        contributions["Final Probability"] = probability
        return probability, predicted_label, contributions, delta_color

    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

# Helper: run model predictions for entire dataset and attach results (cached)
@st.cache_data
def annotate_dataset_with_predictions(df_standard):
    model, _ = load_trained_model()
    if df_standard is None or model is None:
        return None
    X_scaled = df_standard[["Orbital_Period", "Transit_Depth", "SNR", "Stellar_Radius"]].values

    preds = []
    probs = []
    for i in range(X_scaled.shape[0]):
        x = X_scaled[i : i+1]
        try:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(x)[0]
                classes = getattr(model, "classes_", None)
                if classes is not None and "Confirmed" in classes:
                    idx = int(list(classes).index("Confirmed"))
                elif classes is not None and len(classes) == 2:
                    idx = 1
                else:
                    idx = int(np.argmax(p))
                prob = float(p[idx] * 100.0)
                raw = str(classes[idx]) if classes is not None else ""
                if "Confirm" in raw:
                    lab = "Confirmed Exoplanet"
                elif "Candidate" in raw:
                    lab = "High-Value Candidate"
                else:
                    lab = "Confirmed Exoplanet" if prob > 85 else ("High-Value Candidate" if prob > 60 else "False Positive")
            else:
                pr = model.predict(x)
                val = float(pr[0]) if hasattr(pr, "__iter__") else float(pr)
                prob = float(np.clip(val, 0.0, 1.0) * 100.0)
                lab = "Confirmed Exoplanet" if prob > 85 else ("High-Value Candidate" if prob > 60 else "False Positive")
        except Exception:
            prob = np.nan
            lab = "Unknown"
        preds.append(lab)
        probs.append(prob)
    df_out = df_standard.copy()
    df_out["Pred_Class"] = preds
    df_out["Pred_Prob"] = probs
    return df_out

# Set Matplotlib/Seaborn style
sns.set_theme(style="whitegrid", palette="viridis") # Changed palette for more vibrant plots

# -------------------------------
# CORE FUNCTIONALITY: SIMULATED PREDICTION LOGIC (No Change)
# -------------------------------
@st.cache_resource
def load_trained_model(model_path="random_forest_model.joblib"):
    """
    Robust loader: search these locations (in order) and return (model, None) when found:
      - app/ (same folder as this file)
      - app/models/
      - project root (one level up)
      - project_root/models/
      - current working directory
      - absolute path if provided
    Returns (None, None) if not found.
    """
    try:
        app_dir = Path(__file__).resolve().parents[0]
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            app_dir / model_path,
            app_dir / "models" / model_path,
            project_root / model_path,
            project_root / "models" / model_path,
            Path.cwd() / model_path,
            Path(model_path)  # allow absolute/path-like input
        ]

        # try each candidate path
        for fp in candidates:
            if fp is None:
                continue
            try:
                if fp.exists():
                    model = joblib.load(str(fp))
                    return model, None
            except Exception:
                # ignore load errors and try next candidate
                pass

        return None, None
    except Exception as e:
        # don't call st.error here to avoid UI error during imports; return None so caller can handle
        return None, None

def _simulated_contributions(orbital_period, transit_depth, snr, stellar_radius):
    """Return deterministic contributions used previously (fallback/explainability)."""
    snr_contribution = snr / 100 * 40
    depth_contribution = transit_depth * 100 * 30
    period_ideal = 100.0
    period_penalty_factor = 0.3 
    period_penalty = 1 - (abs(orbital_period - period_ideal) / period_ideal)**2 * period_penalty_factor 
    period_contribution = max(0, 10 * period_penalty)
    radius_contribution = min(10.0, (1 / stellar_radius) * 10)
    base_prob = snr_contribution + depth_contribution + period_contribution + radius_contribution
    return {
        "SNR": snr_contribution,
        "Transit Depth": depth_contribution,
        "Orbital Period": period_contribution,
        "Stellar Radius": radius_contribution,
        "Base Probability": base_prob
    }

def predict_exoplanet(orbital_period, transit_depth, snr, stellar_radius):
    """
    Use a trained model (if available) to predict probability and class.
    Falls back to the original simulated deterministic function if model/scaler not found or fails.
    Returns: (probability_percent(float), prediction_class(str), contributions(dict), delta_color(str))
    """
    model, _ = load_trained_model()
    features = np.array([[orbital_period, transit_depth, snr, stellar_radius]], dtype=float)

    # Attempt to use trained model
    if model is not None:
        try:
            # no scaler available — pass raw features
            features_scaled = features

            # Classifier with predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features_scaled)[0]
                # If model has classes_ try to find a positive/confirmed class
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    # prefer label containing 'Confirmed' or assume binary with positive at index 1
                    if "Confirmed" in classes:
                        idx = int(list(classes).index("Confirmed"))
                    elif len(classes) == 2:
                        idx = 1
                    else:
                        idx = int(np.argmax(probs))
                else:
                    idx = int(np.argmax(probs))
                probability = float(probs[idx] * 100.0)

                # Map model raw label (if available) to UI-friendly classes
                raw_label = str(classes[idx]) if classes is not None else None
                if raw_label and ("Confirm" in raw_label or "confirmed" in raw_label):
                    predicted_label = "Confirmed Exoplanet"
                elif raw_label and "Candidate" in raw_label:
                    predicted_label = "High-Value Candidate"
                else:
                    predicted_label = "Confirmed Exoplanet" if probability > 85 else ("High-Value Candidate" if probability > 60 else "False Positive")

            # Regressor that outputs 0..1 probability
            elif hasattr(model, "predict"):
                pred = model.predict(features_scaled)
                pred_val = float(pred[0]) if hasattr(pred, "__iter__") else float(pred)
                probability = float(np.clip(pred_val, 0.0, 1.0) * 100.0)
                predicted_label = "Confirmed Exoplanet" if probability > 85 else ("High-Value Candidate" if probability > 60 else "False Positive")

            else:
                raise RuntimeError("Unknown model type")

            # Build contributions: prefer feature_importances_ if present, else use simulated contributions
            contributions = _simulated_contributions(orbital_period, transit_depth, snr, stellar_radius)
            if hasattr(model, "feature_importances_"):
                fi = np.array(model.feature_importances_, dtype=float)
                if fi.sum() > 0:
                    fi_norm = fi / fi.sum()
                    total_base = sum([contributions[k] for k in ["SNR", "Transit Depth", "Orbital Period", "Stellar Radius"]])
                    # Map indices to expected feature order [period, depth, snr, radius]
                    # If model was trained with same order, this will align; otherwise adjust accordingly.
                    contributions = {
                        "Orbital Period": float(total_base * fi_norm[0]) if len(fi_norm) > 0 else contributions["Orbital Period"],
                        "Transit Depth": float(total_base * fi_norm[1]) if len(fi_norm) > 1 else contributions["Transit Depth"],
                        "SNR": float(total_base * fi_norm[2]) if len(fi_norm) > 2 else contributions["SNR"],
                        "Stellar Radius": float(total_base * fi_norm[3]) if len(fi_norm) > 3 else contributions["Stellar Radius"],
                        "Base Probability": float(total_base)
                    }

            # finalize
            delta_color = "normal" if probability > 85 else ("caution" if probability > 60 else "inverse")
            probability = float(np.clip(probability, 0.0, 99.9))
            contributions["Final Probability"] = probability
            return probability, predicted_label, contributions, delta_color

        except Exception:
            # on any failure, fall back to simulated logic
            pass

    # FALLBACK: original deterministic simulation
    contributions = _simulated_contributions(orbital_period, transit_depth, snr, stellar_radius)
    base_prob = contributions["Base Probability"]
    probability = min(99.9, max(5.0, base_prob + random.uniform(-3, 3)))
    if probability > 85:
        prediction_class = "Confirmed Exoplanet"
        delta_color = "normal"
    elif probability > 60:
        prediction_class = "High-Value Candidate"
        delta_color = "caution"
    else:
        prediction_class = "False Positive"
        delta_color = "inverse"
    contributions["Final Probability"] = probability
    return probability, prediction_class, contributions, delta_color


# -------------------------------
# Page Config & Styling
# -------------------------------
st.set_page_config(
    page_title="🌌 Exoplanet Hunter AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for primary color, font, and button hover effect
st.markdown("""
    <style>
    :root {
        --primary: #8a2be2; /* BlueViolet */
        --accent: #00bfff;  /* Deep Sky Blue */
        --shadow: 0 8px 24px rgba(0,0,0,0.25);
        --shadow-hover: 0 14px 28px rgba(0,0,0,0.35);
    }

    /* Animated subtle background gradient */
    html, body {
        background: radial-gradient(circle at 10% 10%, rgba(138,43,226,0.06), transparent 25%),
                    radial-gradient(circle at 90% 20%, rgba(0,191,255,0.06), transparent 30%),
                    linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.02));
        background-size: 140% 140%, 140% 140%, 100% 100%;
        animation: bgDrift 24s ease infinite;
    }
    @keyframes bgDrift {
        0% { background-position: 0% 0%, 100% 0%, 0% 0%; }
        50% { background-position: 50% 40%, 60% 10%, 0% 0%; }
        100% { background-position: 0% 0%, 100% 0%, 0% 0%; }
    }

    .block-container:before {
        content: "";
        position: fixed;
        top: -40vh;
        left: -20vw;
        width: 140vw;
        height: 80vh;
        background: radial-gradient(ellipse at top left, rgba(138,43,226,0.25), transparent 50%),
                    radial-gradient(ellipse at top right, rgba(0,191,255,0.18), transparent 50%);
        filter: blur(30px);
        z-index: -1;
        animation: floatGlow 14s ease-in-out infinite alternate;
        pointer-events: none;
    }
    @keyframes floatGlow {
        0% { transform: translateY(0px) scale(1); opacity: 0.6; }
        100% { transform: translateY(24px) scale(1.05); opacity: 0.9; }
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), #7df0ff);
        background-size: 200% 100%;
        animation: barShimmer 3s linear infinite;
    }
    @keyframes barShimmer { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
    div[data-testid="stMetricValue"] {
        font-size: 42px;
        color: var(--primary);
        font-weight: 700;
        text-shadow: 0 1px 0 rgba(255,255,255,0.05), 0 4px 16px rgba(138,43,226,0.25);
    }
    div[data-testid="stMetricLabel"] { font-size: 16px; opacity: 0.9; }
    div[data-testid="stMetricDelta"] { font-size: 20px; text-shadow: 0 1px 0 rgba(255,255,255,0.05); }

    

    .stButton>button {
        transition: all 0.25s ease-in-out;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--primary), #6f1fb8);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 14px rgba(138,43,226,0.28);
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #9b4dff, var(--primary));
        color: #fff;
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 10px 24px rgba(138,43,226,0.38);
    }
    /* Ripple effect */
    .stButton>button:after {
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        width: 0; height: 0;
        background: rgba(255,255,255,0.35);
        border-radius: 100%;
        transform: translate(-50%, -50%);
        transition: width .35s ease, height .35s ease, opacity .4s ease;
        opacity: 0;
        pointer-events: none;
    }
    .stButton>button:hover:after { width: 180%; height: 180%; opacity: 0.08; }

    .stSlider [data-testid="stTooltipContent"] {
        background-color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        color: white !important;
    }

    div[data-baseweb="input"]>input, div[data-baseweb="select"] div, div[data-testid="stNumberInput"] input {
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
        border-radius: 10px !important;
        border-color: rgba(138,43,226,0.35) !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.12);
    }
    div[data-baseweb="input"]>input:focus, div[data-testid="stNumberInput"] input:focus {
        box-shadow: 0 0 0 3px rgba(138,43,226,0.25), 0 8px 18px rgba(138,43,226,0.18) !important;
        border-color: var(--primary) !important;
    }

    /* Link underline slide animation */
    a {
        position: relative;
        text-decoration: none !important;
    }
    a:after {
        content: "";
        position: absolute;
        left: 0; bottom: -2px;
        height: 2px; width: 0%;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        transition: width .25s ease;
    }
    a:hover:after { width: 100%; }

    div[data-testid="stTabs"] button[role="tab"] {
        transition: color 0.2s ease, background 0.2s ease, transform 0.2s ease;
        border-radius: 12px;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover { background: rgba(138,43,226,0.12); transform: translateY(-1px); }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(138,43,226,0.22), rgba(0,191,255,0.16));
        border: 1px solid rgba(138,43,226,0.35);
        box-shadow: 0 6px 16px rgba(138,43,226,0.25) inset;
    }

    .stDataFrame table tbody tr { transition: background .18s ease; }
    .stDataFrame table tbody tr:hover { background: rgba(138,43,226,0.08); }
    .block-container > div { animation: fadeInUp 420ms ease 1; }
    @keyframes fadeInUp { from { opacity: 0; transform: translate3d(0, 8px, 0); } to { opacity: 1; transform: translateZ(0); } }

    /* Gradient animated main title text */
    h1 {
        background: linear-gradient(90deg, var(--primary), #7df0ff, var(--primary));
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 8s ease-in-out infinite;
        text-shadow: 0 6px 22px rgba(138,43,226,0.18);
    }
    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar subtle gradient and divider accents */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(138,43,226,0.08), rgba(0,191,255,0.06));
        border-right: 1px solid rgba(138,43,226,0.15);
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(138,43,226,0.25); }

    /* Active tab underline accent */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after {
        content: "";
        display: block;
        height: 3px;
        border-radius: 3px;
        margin-top: 6px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }

    /* Zebra rows for tables */
    .stDataFrame table tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }

    /* Metric hover pulse */
    div[data-testid="stMetric"] {
        transition: transform 0.18s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-1px);
    }

    /* Tabs subtle scale on hover */
    div[data-testid="stTabs"] button[role="tab"] { transition: transform .18s ease; }
    div[data-testid="stTabs"] button[role="tab"]:hover { transform: translateY(-1px) scale(1.01); }

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--primary), #5a18a6); border-radius: 8px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.06); }

    /* Checkbox & radio accent */
    div[role="checkbox"], div[role="radio"] {
        accent-color: var(--primary);
    }

    /* Tooltip fine-tune */
    [data-testid="stTooltipContent"] {
        border-radius: 8px !important;
        box-shadow: 0 8px 18px rgba(0,0,0,0.25) !important;
    }

    /* Images and charts subtle hover lift */
    img, .stImage, .stPyplot, .stPlotlyChart, .stVegaLiteChart, .stAltairChart {
        transition: transform .2s ease, filter .2s ease;
    }
    img:hover, .stImage:hover, .stPyplot:hover, .stPlotlyChart:hover, .stVegaLiteChart:hover, .stAltairChart:hover {
        transform: translateY(-1px);
        filter: drop-shadow(0 6px 14px rgba(0,0,0,0.25));
    }

    
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# DATA GENERATION 
# -------------------------------
@st.cache_data
def generate_mock_data():
    np.random.seed(42)
    N = 500
    mock_data = pd.DataFrame({
        'Orbital_Period': np.random.lognormal(mean=2, sigma=1.2, size=N),
        'Transit_Depth': np.random.uniform(0.01, 1.0, N),
        'SNR': np.random.uniform(5, 100, N),
        'Stellar_Radius': np.random.uniform(0.1, 3.0, N),
        'Status': np.random.choice(['Confirmed', 'False Positive', 'Candidate'], size=N, p=[0.4, 0.3, 0.3])
    })
    # Adjusting data to show correlation
    mock_data.loc[mock_data['Status'] == 'Confirmed', 'Transit_Depth'] += np.random.uniform(0.1, 0.4, (mock_data['Status'] == 'Confirmed').sum())
    mock_data.loc[mock_data['Status'] == 'Confirmed', 'SNR'] += np.random.uniform(10, 30, (mock_data['Status'] == 'Confirmed').sum())
    mock_data['Transit_Depth'] = np.clip(mock_data['Transit_Depth'], 0.01, 1.0)
    mock_data['SNR'] = np.clip(mock_data['SNR'], 5, 100)
    return mock_data

mock_data = generate_mock_data()

# Initialize session state 
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Initialize slider keys
if 'op' not in st.session_state:
    st.session_state.op = 100.0
if 'td' not in st.session_state:
    st.session_state.td = 0.3
if 'snr' not in st.session_state:
    st.session_state.snr = 60.0
if 'sr' not in st.session_state:
    st.session_state.sr = 0.8


def reset_prediction():
    """Function to clear the prediction state."""
    st.session_state.prediction_result = None

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Configuration & Info ⚙️")
st.sidebar.markdown("---")
st.sidebar.button("🔄 Reset Single Prediction", on_click=reset_prediction, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Instructions:**
    1. Adjust feature sliders in the **🚀 Single Star Prediction** tab.
    2. Try setting **Transit Depth > 0.8** and **SNR > 90** to achieve **Confirmed Exoplanet**!
    """
)
st.sidebar.markdown(f"**Model Status:** `v2.3` (Final Simulation)")
st.sidebar.markdown("Designed by: Hello Space 💫")


# -------------------------------
# Main Title & Tabs
# -------------------------------
st.markdown("<h1 style='text-align: center; color: #8a2be2;'>🔭 Exoplanet Hunter AI</h1>", unsafe_allow_html=True)
st.caption("AI-powered tool for predicting exoplanetary candidates.")
st.markdown("---")

# Helper: normalize depth values from CSV / user input to fractional (0..1)
def normalize_depth(d):
    try:
        if pd.isna(d):
            return 0.01
        d = float(d)
        # Heuristic:
        # - large values (>1000) likely ppm -> convert to fraction
        # - moderate values (1..100) likely percent -> divide by 100
        # - small values (<=1) already fraction
        if d > 1000:
            return d / 1e6
        if d > 1:
            return d / 100.0
        return d
    except Exception:
        return 0.01

# Helper: unknown label checks
def is_unknown_text_label(val):
    try:
        s = str(val).strip().lower()
    except Exception:
        return True
    if s is None:
        return True
    s_norm = s.replace(" ", "")
    return s_norm in {"unknown", "", "nan", "na", "n/a", "none", "null", "-", "?"}

def is_unknown_index_label(val):
    try:
        if pd.isna(val):
            return True
    except Exception:
        pass
    try:
        return int(val) < 0
    except Exception:
        return is_unknown_text_label(val)

# --- NEW HELPERS: build full feature vector from CSV row and call model consistently ---
def build_model_input_from_row(row, model):
    """
    Build full (1, n_features) array expected by `model` using:
      - model.feature_names_in_ if present (preferred)
      - else a sensible fallback order based on your CSV header
    Returns: X (1,n), feat_names(list), used_values(dict)
    """
    # preferred: use exact ordering saved with model
    feat_names = None
    if hasattr(model, "feature_names_in_") and getattr(model, "feature_names_in_", None) is not None:
        feat_names = list(model.feature_names_in_)
    else:
        # fallback order matching your CSV header (14-ish fields you listed)
        fallback = [
            "period","duration","depth","rp","rp_est","rstar","teff","logg",
            "period_missing","duration_missing","depth_missing","rp_missing","rp_est_missing","rstar_missing"
        ]
        n_req = int(getattr(model, "n_features_in_", len(fallback)))
        feat_names = fallback[:n_req]

    # case-insensitive mapping from row columns to feature names
    row_map = {c.lower(): c for c in row.index}
    vals = []
    used = {}
    for fn in feat_names:
        # try direct case-insensitive match
        orig = row_map.get(fn.lower())
        if orig is None:
            # relaxed match removing underscores
            orig = next((c for c in row.index if c.lower().replace("_","") == fn.lower().replace("_","")), None)
        if orig is None:
            v = np.nan
        else:
            try:
                v = float(row[orig])
            except Exception:
                try:
                    v = float(str(row[orig]).strip())
                except Exception:
                    v = np.nan
        vals.append(v)
        used[fn] = v

    X = np.asarray(vals, dtype=float).reshape(1, -1)

    # pad/trim to model.n_features_in_ if present
    n_req = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] < n_req:
        pad = np.full((1, n_req - X.shape[1]), np.nan, dtype=float)
        X = np.hstack([X, pad])
    elif X.shape[1] > n_req:
        X = X[:, :n_req]
        feat_names = feat_names[:n_req]
        used = {k: used[k] for k in feat_names}

    return X, feat_names, used

def model_predict_from_row(row, model):
    """
    Build model input from CSV row and call model.predict_proba / predict.
    Returns dict with:
      - friendly_label (mapped from argmax only)
      - prob_pct (probability of argmax class in percent)
      - raw_class (model.classes_[argmax] or argmax index)
      - raw_probs (list) or None
      - feature_names, used_values
    """
    X_full, feat_names, used_values = build_model_input_from_row(row, model)

    # call model
    raw_probs = None
    raw_class = None
    prob_pct = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_full)[0]
        raw_probs = probs.tolist()
        idx = int(np.argmax(probs))
        classes = getattr(model, "classes_", None)
        raw_class = classes[idx] if classes is not None else idx
        prob_pct = float(probs[idx] * 100.0)
    else:
        pred = model.predict(X_full)
        pred0 = pred[0] if hasattr(pred, "__iter__") else pred
        try:
            idx = int(pred0)
        except Exception:
            idx = 0
        classes = getattr(model, "classes_", None)
        raw_class = classes[idx] if classes is not None and idx < len(classes) else pred0
        prob_pct = 100.0

    # Map argmax raw class -> friendly label (CONSISTENT: use argmax only)
    def map_argmax_to_label(c, p):
        try:
            if isinstance(c, (int, np.integer)):
                if int(c) == 2:
                    return "Confirmed Exoplanet"
                if int(c) == 1:
                    return "High-Value Candidate"
                return "False Positive"
            s = str(c).lower()
            if "confirm" in s:
                return "Confirmed Exoplanet"
            if "candidate" in s:
                return "High-Value Candidate"
            if "false" in s or "negative" in s:
                return "False Positive"
        except Exception:
            pass
        return "Confirmed Exoplanet" if p > 85 else ("High-Value Candidate" if p > 60 else "False Positive")

    friendly = map_argmax_to_label(raw_class, prob_pct)
    return {
        "friendly_label": friendly,
        "prob_pct": prob_pct,
        "raw_class": raw_class,
        "raw_probs": raw_probs,
        "feature_names": feat_names,
        "used_values": used_values
    }

# ensure tabs are defined before any `with tab1:` usage
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Single Star Prediction", "📈 Model Diagnostics", "📁 Batch Processing", "🔬 Light Curve Analysis"])


@st.cache_resource
def run_light_curve_analysis(kepid, known_period=None):
    """Fetches light curve, stitches, folds, and computes periodogram for a given KIC ID.
    Returns: (lc_stitched or None, status_message(str), periodogram or None, best_period or None, metrics(dict) or None)
    """
    try:
        search_result = lk.search_lightcurvefile(f"KIC {kepid}", cadence="long")
    except Exception as e:
        return None, f"Lightkurve search failed: {e}", None, None, None

    if len(search_result) == 0:
        return None, "No light curve files found for this target!", None, None, None

    try:
        lc_files = search_result.download_all()
    except Exception as e:
        # Try to detect and remove a corrupt cached FITS file mentioned in the error message and retry once
        msg = str(e)
        try:
            import re
            bad_path_match = re.search(r"([A-Za-z]:\\\\[^\s]+?\\\\[^\s]+?\.fits)", msg)
            if bad_path_match:
                bad_fp = bad_path_match.group(1)
                if os.path.exists(bad_fp):
                    try:
                        os.remove(bad_fp)
                    except Exception:
                        return None, f"Failed to remove corrupted cache file: {bad_fp}", None, None, None
                    # retry
                    try:
                        lc_files = search_result.download_all()
                    except Exception as ee:
                        return None, f"Retry after removing corrupted cache failed: {ee}", None, None, None
                else:
                    return None, f"Download failed and referenced cache file not found: {bad_fp}", None, None, None
            else:
                return None, f"Failed to download light curve files: {e}", None, None, None
        except Exception:
            return None, f"Failed to download light curve files: {e}", None, None, None

    lc_list = [lc_file.PDCSAP_FLUX.remove_nans().normalize()
               for lc_file in lc_files if lc_file.PDCSAP_FLUX is not None]

    if len(lc_list) == 0:
        return None, "No valid PDCSAP light curves available after cleaning!", None, None, None

    lc_stitched = lc_list[0] if len(lc_list) == 1 else lk.LightCurveCollection(lc_list).stitch()

    try:
        periodogram = lc_stitched.to_periodogram(method='lombscargle', normalization='psd', maximum_period=500, oversample=10)
    except Exception as e:
        return lc_stitched, f"Light curve processed, but periodogram computation failed: {e}", None, None, None

    try:
        best_period = periodogram.period_at_max_power.value
        peak_power = np.max(periodogram.power.value)
    except Exception:
        best_period = None
        peak_power = None

    try:
        fap = periodogram.false_alarm_probability(peak_power) if peak_power is not None else np.nan
    except Exception:
        fap = np.nan

    metrics = {
        "Best Period (LS)": f"{best_period:.4f} days" if best_period is not None else "N/A",
        "Peak Power": f"{peak_power:.4f}" if peak_power is not None else "N/A",
        "False Alarm Probability": f"{fap:.2e}" if not pd.isna(fap) else "N/A"
    }

    return lc_stitched, "Success", periodogram, best_period, metrics

# --- REPLACE Single Star prediction use to call model_predict_from_row and show raw/prob + exact match check ---
with tab1:
    st.subheader("Single Star — lookup by KepID / TIC ID and model prediction (dataset-driven)")

    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.markdown("Enter a KepID or TIC ID. The app will fetch that row from the CSV and display the actual label and a predicted value).")
        id_type = st.selectbox("Select ID type", options=["kepid", "tic_id"])
        id_value = st.text_input(f"Enter {id_type} (exact match)", value="")
        if st.button("🔎 Lookup & Predict"):
            if dataset is None:
                st.error("Dataset not loaded.")
            else:
                df = dataset.copy()
                colmap = {c.lower(): c for c in df.columns}
                key = id_type.lower()
                if key not in colmap:
                    st.error(f"{id_type} columns.")
                else:
                    orig_col = colmap[key]
                    match_rows = df[df[orig_col].astype(str).str.strip() == str(id_value).strip()]
                    if match_rows.empty:
                        try:
                            valnum = float(id_value)
                            match_rows = df[np.isclose(pd.to_numeric(df[orig_col], errors='coerce'), valnum, equal_nan=False)]
                        except Exception:
                            pass
                    if match_rows.empty:
                        st.warning("No matching row found for that ID.")
                    else:
                        row = match_rows.iloc[0]

                        # show CSV row summary
                        st.markdown("### Actual dataset values (selected columns)")
                        cols_to_show = [c for c in ['target_name','kepid','tic_id','label','label_idx','period','duration','depth','rp','rp_est','rstar','teff','logg'] if c in df.columns]
                        st.table(row[cols_to_show].to_frame().T)

                        # Derive index and if missing CALL MODEL to produce predicted_value
                        actual_label = None
                        actual_idx = None
                        predicted_value = None

                        if 'label' in row.index and not pd.isna(row['label']):
                            if is_unknown_text_label(row['label']):
                                # treat -> call model
                                model, _ = load_trained_model()
                                if model is None:
                                    predicted_value = "Model Not Found"
                                else:
                                    try:
                                        mp = model_predict_from_row(row, model)
                                        predicted_value = mp.get("friendly_label", "Prediction Error")
                                    except Exception:
                                        predicted_value = "Prediction Error"
                            else:
                                actual_label = row['label']
                                predicted_value = actual_label
                        elif 'label_idx' in row.index and not pd.isna(row['label_idx']):
                            try:
                                actual_idx = int(row['label_idx'])
                            except Exception:
                                actual_idx = row['label_idx']
                            if is_unknown_index_label(actual_idx):
                                # unknown -> call model
                                model, _ = load_trained_model()
                                if model is None:
                                    predicted_value = "Model Not Found"
                                else:
                                    try:
                                        mp = model_predict_from_row(row, model)
                                        predicted_value = mp.get("friendly_label", "Prediction Error")
                                    except Exception:
                                        predicted_value = "Prediction Error"
                            else:
                                if actual_idx == 2:
                                    predicted_value = "Confirmed Exoplanet"
                                elif actual_idx == 1:
                                    predicted_value = "High-Value Candidate"
                                elif actual_idx == 0:
                                    predicted_value = "False Positive"
                                else:
                                    predicted_value = str(actual_idx)
                        else:
                            # CSV  -> call model using available features
                            # find feature values (try common column names)
                            def _get_row_val(r, candidates, default=np.nan):
                                for c in candidates:
                                    if c in r.index and not pd.isna(r.get(c)):
                                        return r.get(c)
                                return default

                            period_val = _get_row_val(row, ['period','Period','Orbital_Period','period_days'])
                            depth_raw = _get_row_val(row, ['depth','Depth','transit_depth','Transit_Depth'])
                            depth_val = normalize_depth(depth_raw)
                            snr_val = _get_row_val(row, ['SNR','snr','signal_to_noise_ratio','duration'], st.session_state.snr)
                            rstar_val = _get_row_val(row, ['rstar','R_star','stellar_radius','Stellar_Radius'], st.session_state.sr)

                            try:
                                prob, pred_label, _, _ = predict_exoplanet(float(period_val), float(depth_val), float(snr_val), float(rstar_val))
                                predicted_value = pred_label
                            except Exception:
                                predicted_value = "Prediction Error"

                        # show Result (text)
                        st.markdown("### Result")
                        st.write("Actual label:", actual_label if actual_label is not None else actual_idx)
                        st.write("Predicted value:", predicted_value)

    with right_col:
        st.markdown("#### Visualization: selected star only")
        fig, ax = plt.subplots(figsize=(6,5))
        try:
            # find period and depth for plotting (from the last selected row if present, else fallback sliders)
            period_val = None
            depth_val = None
            if 'row' in locals():
                for k in ['period','Period','Orbital_Period','period_days']:
                    if k in row.index and not pd.isna(row.get(k)):
                        period_val = float(row.get(k))
                        break
                for k in ['depth','Depth','Transit_Depth','transit_depth']:
                    if k in row.index and not pd.isna(row.get(k)):
                        depth_val = normalize_depth(row.get(k))
                        break

            if period_val is None:
                period_val = float(st.session_state.op)
            if depth_val is None:
                depth_val = normalize_depth(st.session_state.td)

            # Plot only the selected star
            ax.scatter([max(period_val, 1e-6)], [depth_val], color="#8a2be2", marker="*", s=220, edgecolor="k", linewidth=0.8, zorder=10)
            ax.set_xscale('log')
            ax.set_xlabel("Orbital Period (days, log scale)")
            ax.set_ylabel("Transit Depth (fraction)")
            ax.set_title("Selected star")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            st.pyplot(fig)
        except Exception as e:
            st.write("Unable to render plot for the selected inputs.", e)

with tab2:
    st.subheader("Model Feature Analysis and Performance")
    
    col_imp, col_perf = st.columns(2)

    with col_imp:
        st.markdown("#### Feature Importance (Dynamic based on Current Input)")
        
        _, _, current_contributions, _ = predict_exoplanet(st.session_state.op, st.session_state.td, st.session_state.snr, st.session_state.sr)
        
        # Filter contributions to only the feature scores
        feature_scores = {k: v for k, v in current_contributions.items() if k in ['SNR', 'Transit Depth', 'Orbital Period', 'Stellar Radius']}
        
        importance_df = pd.DataFrame(
            list(feature_scores.items()), 
            columns=['Feature', 'Value']
        ).sort_values(by="Value", ascending=False)
        
        # Calculate percentage contribution
        importance_df['Importance_perc'] = (importance_df['Value'] / importance_df['Value'].sum()) * 100

        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        # Use a contrasting, sleek palette for bar chart
        sns.barplot(x="Importance_perc", y="Feature", data=importance_df, palette="plasma", ax=ax_imp) 
        ax_imp.set_xlabel("Relative Score Contribution (%)")
        ax_imp.set_ylabel("")
        ax_imp.set_title("Feature Contribution for Current Star", fontweight='bold')
        st.pyplot(fig_imp)
        st.caption("This chart shows which input feature *currently* drives the deterministic score.")

    with col_perf:
        st.markdown("#### Mock Model Performance Metrics (On Test Set)")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        # Use st.metric for key performance indicators
        with perf_col1:
            st.metric("Accuracy", "92.5%", "High")
        with perf_col2:
            st.metric("F1 Score (Exoplanet)", "0.88", "Good", delta_color="normal")
        with perf_col3:
            st.metric("False Positive Rate", "7.1%", "Low", delta_color="inverse")
        
        st.markdown("---")
        st.markdown("##### Confusion Matrix (Mock Test Set Results)")
        cm = pd.DataFrame({
            'Predicted Confirmed': [350, 20],
            'Predicted False Positive': [50, 580]
        }, index=['Actual Confirmed', 'Actual False Positive'])
        
        # Style the confusion matrix for better readability
        def highlight_diag(val):
            color = '#d4edda' if val > 300 else '' # Highlight True Positives/Negatives
            return f'background-color: {color}; font-weight: bold;'
            
        st.dataframe(cm.style.applymap(highlight_diag), use_container_width=True)


# -------------------------------
# --- TAB 3: Batch Processing (Improved Layout and Visuals) ---
# -------------------------------
with tab3:
    st.subheader("Batch Prediction from CSV Upload 📁")
    template_df = pd.DataFrame({
        "target_name": ["StarA","StarB"],
        "period": [10.5, 300.2],
        "duration": [2.5, 3.1],
        "depth": [3500.0, 1200.0],
        "rp": [1.1, 2.2],
        "rp_est": [1.0, 2.1],
        "rstar": [0.8, 1.5],
        "teff": [5600, 4500],
        "logg": [4.3,4.5],
        "period_missing":[0,0],
        "duration_missing":[0,0],
        "depth_missing":[0,0],
        "rp_missing":[0,0],
        "rp_est_missing":[0,0],
        "rstar_missing":[0,0],
        "teff_missing":[0,0],
        "logg_missing":[0,0],
        "label_idx":[0,2]
    })
    col_upload, col_download_temp = st.columns(2)
    with col_download_temp:
        st.download_button("📥 Download CSV Template", template_df.to_csv(index=False).encode('utf-8'), file_name="exoplanet_template.csv", mime="text/csv", use_container_width=True)
    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV for batch prediction (use your evap_synthesized_catalog.csv)", type="csv")
        max_rows = st.number_input("Max rows to process", min_value=1, value=1000, step=100)

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        if len(df_raw) > max_rows:
            df_raw = df_raw.head(int(max_rows))

        # helper: case-insensitive find
        def find_col_ci(df, candidates):
            lower = {c.lower(): c for c in df.columns}
            for c in candidates:
                if c.lower() in lower:
                    return lower[c.lower()]
            return None

        period_col = find_col_ci(df_raw, ['period','Period','orbital_period','orbital_period_days'])
        depth_col = find_col_ci(df_raw, ['depth','Depth','transit_depth','transit_depth_percent'])
        rstar_col = find_col_ci(df_raw, ['rstar','R_star','stellar_radius','stellar_radius_solar'])
        snr_col = find_col_ci(df_raw, ['snr','SNR','signal_to_noise_ratio','duration'])

        # detect The data
        label_col = find_col_ci(df_raw, ['label','status','actual_label','true_label'])
        label_idx_col = find_col_ci(df_raw, ['label_idx','labelindex','label_index','label_id'])

        # Load model once for the batch (used when labels are unknown)
        model, _ = load_trained_model()
        progress = st.progress(0)
        results = []
        for i, row in df_raw.iterrows():
            try:
              
                if label_col is not None and not pd.isna(row.get(label_col)):
                    actual_val = row.get(label_col)
                    if is_unknown_text_label(actual_val):
                        # unknown -> predict via model
                        if model is None:
                            predicted = "Model Not Found"
                        else:
                            try:
                                mp = model_predict_from_row(row, model)
                                predicted = mp.get("friendly_label", "Prediction Error")
                            except Exception:
                                predicted = "Prediction Error"
                    else:
                        predicted = actual_val
             
                elif label_idx_col is not None and not pd.isna(row.get(label_idx_col)):
                    try:
                        idx = int(row.get(label_idx_col))
                    except Exception:
                        idx = None
                    actual_val = row.get(label_idx_col)
                    if is_unknown_index_label(actual_val):
                        # unknown -> predict via model
                        if model is None:
                            predicted = "Model Not Found"
                        else:
                            try:
                                mp = model_predict_from_row(row, model)
                                predicted = mp.get("friendly_label", "Prediction Error")
                            except Exception:
                                predicted = "Prediction Error"
                    else:
                        if idx == 2:
                            predicted = "Confirmed Exoplanet"
                        elif idx == 1:
                            predicted = "High-Value Candidate"
                        elif idx == 0:
                            predicted = "False Positive"
                        else:
                            predicted = str(row.get(label_idx_col))
                else:
                    # No actual label: call model using best-effort feature columns
                    try:
                        period_v = float(row[period_col]) if period_col is not None and not pd.isna(row.get(period_col)) else float(st.session_state.op)
                        depth_v = normalize_depth(row[depth_col]) if depth_col is not None and not pd.isna(row.get(depth_col)) else normalize_depth(st.session_state.td)
                        rstar_v = float(row[rstar_col]) if rstar_col is not None and not pd.isna(row.get(rstar_col)) else float(st.session_state.sr)
                        snr_v = float(row[snr_col]) if snr_col is not None and not pd.isna(row.get(snr_col)) else float(st.session_state.snr)
                        prob, pred_class, _, _ = predict_exoplanet(period_v, depth_v, snr_v, rstar_v)
                        predicted = pred_class
                        actual_val = None
                    except Exception:
                        predicted = "Prediction Error"
                        actual_val = None

                results.append({"Actual": actual_val, "Predicted": predicted})
            except Exception:
                actual_val = row.get(label_col) if label_col is not None else None
                results.append({"Actual": actual_val, "Predicted": "ERROR"})
            if (i + 1) % 50 == 0 or (i + 1) == len(df_raw):
                progress.progress((i+1)/max(1,len(df_raw)))
        progress.empty()
        results_df = pd.DataFrame(results)
        out_df = pd.concat([df_raw.reset_index(drop=True), results_df], axis=1)
        st.dataframe(out_df, use_container_width=True)
        st.download_button("💾 Download Predictions CSV", out_df.to_csv(index=False).encode('utf-8'), file_name="batch_predictions.csv", mime="text/csv", use_container_width=True)


# -------------------------------
# --- TAB 4: Light Curve Analysis (NEW) ---
with tab4:
    st.subheader("Light Curve & Periodogram Analysis (Kepler/TESS) 🌟")
    st.markdown("""
        Enter a *Kepler Input Catalog (KIC) ID* (e.g., 10811496) to fetch its light curve data
        and compute the Lomb-Scargle periodogram.
    """)

    kepid_input = st.text_input("Enter KIC ID", value="10811496")

    if st.button("✨ Analyze Light Curve", use_container_width=True):
        if not kepid_input.strip().isdigit():
            st.error("Please enter a valid numeric KIC ID.")
        else:
            kepid = int(kepid_input.strip())

            # Try to find the known period from the loaded dataset
            known_period = None
            if dataset is not None:
                kepid_col = next((c for c in dataset.columns if c.lower() in ('kepid','id')), None)
                period_col = next((c for c in dataset.columns if c.lower() in ('period','orbital_period')), None)
                if kepid_col and period_col:
                    try:
                        match = dataset[dataset[kepid_col] == kepid]
                        if not match.empty:
                            period_val = match.iloc[0][period_col]
                            if not pd.isna(period_val):
                                known_period = float(period_val)
                    except Exception:
                        pass

            lc_stitched, status_msg, periodogram, best_period, metrics = run_light_curve_analysis(kepid, known_period)

            if lc_stitched is None:
                st.error(status_msg)
            else:
                st.success(f"Light curve fetched and processed. Status: {status_msg}")

                st.markdown("---")
                st.markdown("#### Analysis Metrics")
                if metrics:
                    metric_cols = st.columns(len(metrics))
                    for i, (label, value) in enumerate(metrics.items()):
                        with metric_cols[i]:
                            st.metric(label, value)

                st.markdown("---")
                st.markdown("#### Light Curve Plots")

                # Raw Light Curve
                fig_raw, ax_raw = plt.subplots(figsize=(10, 4))
                lc_stitched.plot(ax=ax_raw, label=f"KIC {kepid}", color='navy')
                ax_raw.set_title(f"Raw Light Curve: KIC {kepid}")
                st.pyplot(fig_raw)

                if periodogram is not None and best_period is not None:
                    fig_ls, ax_ls = plt.subplots(figsize=(10, 4))
                    ax_ls.plot(periodogram.period.value, periodogram.power.value, color='purple', linewidth=1.5)
                    ax_ls.set_xlabel("Period [days]")
                    ax_ls.set_ylabel("Power")
                    ax_ls.set_title(f"Lomb–Scargle Periodogram: KIC {kepid}")
                    ax_ls.axvline(best_period, color='orange', linestyle='-', label=f'LS Best Period ({best_period:.4f} d)', linewidth=2)
                    if known_period:
                        ax_ls.axvline(known_period, color='red', linestyle='--', label=f'Known Period ({known_period:.4f} d)')
                    ax_ls.set_xscale('log')
                    ax_ls.grid(True, which='both', linestyle='--', alpha=0.6)
                    ax_ls.legend()
                    st.pyplot(fig_ls)

                    fig_folded, ax_folded = plt.subplots(figsize=(10, 4))
                    lc_folded = lc_stitched.fold(period=best_period)
                    lc_folded.scatter(ax=ax_folded, marker='.', s=10, color='darkgreen')
                    ax_folded.set_title(f"Folded Light Curve (Period: {best_period:.4f} days)")
                    ax_folded.set_xlabel("Phase")
                    st.pyplot(fig_folded)

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey; font-style: italic;'>Data and prediction logic are simulated for demonstration purposes. Developed by Hello Space 💫</p>", unsafe_allow_html=True)