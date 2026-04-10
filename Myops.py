import streamlit as st
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Myopia Insight | Hitha Krishna", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=IBM+Plex+Sans:wght@500;700&family=Lexend:wght@700&display=swap');
    
    .stApp { background-color: #F8FAFC; }
    
    /* Header with Fluorescent Purple Title */
    .header-container {
        background-color: #FFFFFF;
        padding: 1.5rem 2rem;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 2rem;
    }
    .main-title { 
        font-family: 'Lexend', sans-serif; 
        font-size: 38px; 
        font-weight: 700; 
        color: #A855F7; 
        margin: 0;
    }

    /* SLIDERS: Fluorescent Cyan */
    div[data-baseweb="slider"] > div:first-child > div:nth-child(2) {
        background: #22D3EE !important;
    }
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #22D3EE !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 12px #22D3EE !important;
    }

    /* TOGGLE & RADIOS: Fluorescent Purple */
    div[data-testid="stCheckbox"] > label > div[role="switch"] > div[data-checked="true"] {
        background-color: #A855F7 !important;
    }
    div[role="radiogroup"] div[data-checked="true"] > div {
        background-color: #A855F7 !important;
        border-color: #A855F7 !important;
    }

    /* Recommendation Cards */
    .rec-card { padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid; }
    .rec-warning { background-color: #FFFBEB; border-color: #F59E0B; color: #92400E; }
    .rec-success { background-color: #F0FDF4; border-color: #10B981; color: #065F46; }
    </style>
""", unsafe_allow_html=True)

# --- 2. ML ENGINE ---
@st.cache_resource
def train_clinical_model():
    n_samples = 40000 
    age = np.random.uniform(2, 23, n_samples)
    current_rx = np.random.uniform(-8, 0.5, n_samples)
    outdoor = np.random.uniform(0, 5, n_samples)
    screen = np.random.uniform(0, 14, n_samples)
    genetics = np.random.choice([0, 1, 2], n_samples)
    is_preterm = np.random.choice([0, 1], n_samples)
    has_gd = np.random.choice([0, 1], n_samples)
    work_dist = np.random.uniform(15, 60, n_samples)
    onset_age = np.random.uniform(3, 15, n_samples)
    prog = ((-0.8 * (24 - age)) * (1 / (age * 0.5)) + (outdoor * 0.2) - (screen * 0.25) - (genetics * 0.4))
    target = current_rx + prog
    X = pd.DataFrame({'age': age, 'rx': current_rx, 'out': outdoor, 'scr': screen, 
                      'gen': genetics, 'pre': is_preterm, 'gd': has_gd, 'slp': 8, 
                      'sex': 1, 'dist': work_dist, 'onset': onset_age})
    return RandomForestRegressor(n_estimators=100, random_state=42).fit(X, target)

model = train_clinical_model()

# --- 3. HEADER ---
st.markdown("""<div class="header-container"><div class="main-title">Myopia Insight AI</div></div>""", unsafe_allow_html=True)

# --- 4. SIDEBAR (Explicit Gender Re-added) ---
with st.sidebar:
    st.header("Patient Profile")
    age_now = st.slider("Current Age", 2, 23, 10)
    
    # --- GENDER COMPONENT ---
    gender_input = st.radio("Biological Gender", ["Male", "Female", "Other"], horizontal=True)
    
    country = st.selectbox("Country/Region", ["India", "Singapore", "China", "USA", "UK", "Australia", "Other"])
    heritage = st.toggle("East/South Asian Heritage")
    st.divider()
    
    st.subheader("OD (Right Eye)")
    od_sph = st.number_input("OD Sphere", -15.0, 2.0, -1.50, step=0.25)
    od_cyl = st.number_input("OD Cylinder", -8.0, 0.0, -0.50, step=0.25)
    
    st.subheader("OS (Left Eye)")
    os_sph = st.number_input("OS Sphere", -15.0, 2.0, -1.25, step=0.25)
    os_cyl = st.number_input("OS Cylinder", -8.0, 0.0, -0.50, step=0.25)
    st.divider()
    
    st.subheader("History")
    parents = st.radio("Myopic Parents", [0, 1, 2], horizontal=True)
    onset = st.number_input("Onset Age", 2, age_now, 7)
    preterm = st.toggle("Preterm Birth")
    gd = st.toggle("Maternal GD")

# --- 5. MAIN DASHBOARD (Explicit Toggle Re-added) ---
st.subheader("Intervention & Lifestyle")

# --- TREATMENT TOGGLE ---
treatment_toggle = st.toggle("Enable Treatment Simulation Protocol")

c1, c2, c3 = st.columns(3)
with c1: outdoor = st.slider("Outdoor Hours", 0.0, 6.0, 1.0)
with c2: screen = st.slider("Near-Work Hours", 0.0, 16.0, 6.0)
with c3: dist = st.slider("Working Distance (cm)", 10, 60, 35)

# Logic
years = np.arange(age_now, 25)
def get_path(sph, treat):
    path, curr = [], sph
    rate = 0.55 if treat else 1.0
    for y in years:
        f = np.array([[y, curr, outdoor, screen, parents, int(preterm), int(gd), 8, 1, dist, onset]])
        p = model.predict(f)[0]
        curr = curr + ((p - curr) * rate)
        path.append(curr)
    return path

p_baseline = get_path(od_sph, False)
p_treated = get_path(od_sph, True)

chart_col, rec_col = st.columns([2, 1])
with chart_col:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=p_baseline, name='Baseline', line=dict(color='#22D3EE', width=2, dash='dot')))
    if treatment_toggle:
        fig.add_trace(go.Scatter(x=years, y=p_treated, name='Treated', line=dict(color='#A855F7', width=5)))
    fig.update_layout(template="simple_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

with rec_col:
    st.subheader("Analysis")
    if dist < 30:
        st.markdown(f'<div class="rec-card rec-warning"><b>Distance Alert:</b> {dist}cm is too near.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rec-card rec-success"><b>Distance OK.</b></div>', unsafe_allow_html=True)
    
    if outdoor < 2:
        st.markdown('<div class="rec-card rec-warning"><b>Light Alert:</b> Need more sun.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rec-card rec-success"><b>Light OK.</b></div>', unsafe_allow_html=True)
    
    st.metric("Final Rx Forecast", f"{p_treated[-1] if treatment_toggle else p_baseline[-1]:.2f} D")

# --- 6. CITATION & EXPORT ---
st.divider()
st.code(f"""@software{{krishna_myopia_2026,
  author = {{Krishna, Hitha}},
  title = {{Myopia Insight AI}},
  year = {{2026}},
  url = {{https://github.com/Meerarkrish/Myopia-Insight}}
}}""", language="latex")

meta = {"gender": [gender_input], "country": [country], "final_rx": [p_treated[-1] if treatment_toggle else p_baseline[-1]]}
st.download_button("Export Research CSV", pd.DataFrame(meta).to_csv(index=False).encode('utf-8'), "myopia_data.csv")