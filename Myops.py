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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Lexend:wght@700&display=swap');
    
    .stApp { background-color: #F8FAFC; }
    
    /* 1. HEADING: Fluorescent Purple */
    .main-title { 
        font-family: 'Lexend', sans-serif; 
        font-size: 42px; 
        color: #A855F7; 
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* 2. SLIDERS: Fluorescent Green Track & Cursor */
    /* The Track */
    div[data-baseweb="slider"] > div > div {
        background: #39FF14 !important;
    }
    /* The Cursor/Handle */
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #39FF14 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 15px #39FF14 !important;
    }

    /* 3. TOGGLE & RADIOS: Neon Blue */
    /* Toggle Background when ON */
    div[data-testid="stCheckbox"] > label > div[role="switch"] > div[data-checked="true"] {
        background-color: #00FFFF !important;
    }
    /* Radio Button Circle when selected */
    div[role="radiogroup"] div[data-checked="true"] > div {
        background-color: #00FFFF !important;
        border-color: #00FFFF !important;
        box-shadow: 0 0 10px #00FFFF !important;
    }
    
    /* Clean up sidebar and metric cards */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; border: 1px solid #E2E8F0; }
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

# --- 3. MAIN HEADING ---
st.markdown('<h1 class="main-title">Myopia Insight AI</h1>', unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Patient Inputs")
    age_now = st.slider("Current Age", 2, 23, 10)
    
    # GENDER (Restored & Neon Blue Styled)
    gender_choice = st.radio("Biological Gender", ["Male", "Female", "Other"], horizontal=True)
    
    country = st.selectbox("Region", ["India", "Singapore", "China", "USA", "UK", "Australia", "Other"])
    heritage = st.toggle("East/South Asian Heritage")
    
    st.divider()
    st.subheader("Refraction Data")
    od_sph = st.number_input("OD Sphere", -15.0, 2.0, -1.50, step=0.25)
    od_cyl = st.number_input("OD Cylinder", -8.0, 0.0, -0.50, step=0.25)
    os_sph = st.number_input("OS Sphere", -15.0, 2.0, -1.25, step=0.25)
    os_cyl = st.number_input("OS Cylinder", -8.0, 0.0, -0.50, step=0.25)
    
    st.divider()
    st.subheader("Medical History")
    parents = st.radio("Myopic Parents", [0, 1, 2], horizontal=True)
    onset = st.number_input("Onset Age", 2, age_now, 7)
    preterm = st.toggle("Preterm Birth")
    gd = st.toggle("Maternal GD")

# --- 5. MAIN DASHBOARD ---
st.markdown("### Clinical Simulation & Lifestyle")

# TREATMENT TOGGLE (Neon Blue Styled)
treatment_on = st.toggle("Activate Treatment Simulation")

c1, c2, c3 = st.columns(3)
with c1: outdoor = st.slider("Outdoor Sunlight (Hrs)", 0.0, 6.0, 1.0)
with c2: screen = st.slider("Near-Work Hours", 0.0, 16.0, 6.0)
with c3: dist = st.slider("Distance (cm)", 10, 60, 35)

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

path_baseline = get_path(od_sph, False)
path_treated = get_path(od_sph, True)

# --- VISUALS ---
chart_col, stat_col = st.columns([2, 1])

with chart_col:
    fig = go.Figure()
    # Baseline: Purple Dash
    fig.add_trace(go.Scatter(x=years, y=path_baseline, name='Baseline', line=dict(color='#CBD5E1', width=2, dash='dot')))
    
    if treatment_on:
        # 4. TREATMENT LINE: Fluorescent Purple
        fig.add_trace(go.Scatter(x=years, y=path_treated, name='Clinical Path', line=dict(color='#A855F7', width=6)))
    
    fig.update_layout(template="simple_white", height=450, yaxis_title="Sphere (D)", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

with stat_col:
    st.subheader("Analysis")
    final_rx = path_treated[-1] if treatment_on else path_baseline[-1]
    st.metric("Forecasted Rx (Age 24)", f"{final_rx:.2f} D")
    
    if dist < 30:
        st.error(f"⚠️ Proximity Alert: {dist}cm is too near.")
    else:
        st.success("✅ Working distance is healthy.")
        
    if outdoor < 2:
        st.warning("⚠️ Light Alert: Need 2+ hrs sun.")
    else:
        st.success("✅ Sunlight levels optimal.")

# --- 6. CITATION ---
st.divider()
st.markdown("**Academic Citation**")
st.code(f"""@software{{krishna_myopia_2026,
  author = {{Krishna, Hitha}},
  title = {{Myopia Insight AI}},
  year = {{2026}},
  url = {{https://github.com/Meerarkrish/Myopia-Insight}}
}}""", language="latex")

# CSV Export
meta = {"gender": [gender_choice], "final_rx": [final_rx], "treatment": [treatment_on]}
st.download_button("Export Research CSV", pd.DataFrame(meta).to_csv(index=False).encode('utf-8'), "myopia_data.csv")