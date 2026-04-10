import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- 1. PROFESSIONAL UI CONFIGURATION ---
st.set_page_config(page_title="Myopia Insight | Clinical Research Portal", layout="wide")

st.markdown("""
    <style>
    /* Professional Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Lexend:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
    }
    
    h1, h2, h3 {
        font-family: 'Lexend', sans-serif;
        color: #0F172A;
        letter-spacing: -0.02em;
    }

    /* Professional Header Section */
    .header-container {
        background-color: #FFFFFF;
        padding: 40px 60px;
        margin: -6rem -5rem 2rem -5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .main-title { font-size: 32px; font-weight: 700; color: #059669; }
    .sub-title { font-size: 16px; color: #64748B; margin-top: 5px; }

    /* Dashboard Cards */
    .clinical-card {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .intervention-card {
        background: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 16px;
        border-radius: 4px;
        color: #065F46;
        font-size: 14px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. ML CORE ---
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
    sleep = np.random.uniform(4, 13, n_samples)
    gender = np.random.choice([0, 1], n_samples)
    work_dist = np.random.uniform(15, 60, n_samples)
    onset_age = np.random.uniform(3, 15, n_samples)

    prog = ((-0.8 * (24 - age)) * (1 / (age * 0.6)) + (outdoor * 0.2) - (screen * 0.2) - (genetics * 0.4))
    target = current_rx + prog
    X = pd.DataFrame({'age': age, 'rx': current_rx, 'out': outdoor, 'scr': screen, 
                      'gen': genetics, 'pre': is_preterm, 'gd': has_gd, 'slp': sleep, 
                      'sex': gender, 'dist': work_dist, 'onset': onset_age})
    return RandomForestRegressor(n_estimators=100, random_state=42).fit(X, target)

model = train_clinical_model()

# --- 3. HEADER ---
st.markdown("""
    <div class="header-container">
        <div class="main-title">Myopia Insight AI</div>
        <div class="sub-title">Clinical Decision Support System & Open Data Repository</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. INPUT INTERFACE ---
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Patient Parameters")
    age_now = st.slider("Current Age", 2, 23, 10)
    gender = st.selectbox("Biological Sex", ["Male", "Female", "Other"])
    country = st.selectbox("Region", ["East Asia", "South Asia", "North America", "Europe", "Other"])
    onset = st.number_input("Age at Initial Prescription", 2, age_now, 7)

with c2:
    st.subheader("Clinical Data (OD/OS)")
    od_s = st.number_input("Right Eye Sphere (D)", -15.0, 2.0, -1.50, step=0.25)
    os_s = st.number_input("Left Eye Sphere (D)", -15.0, 2.0, -1.25, step=0.25)
    st.write("---")
    preterm = st.checkbox("History of Preterm Birth")
    gd = st.checkbox("Maternal Gestational Diabetes")
    parents = st.radio("Number of Myopic Parents", [0, 1, 2], horizontal=True)

with c3:
    st.subheader("Environmental Factors")
    outdoor = st.slider("Outdoor Light Exposure (Hrs/Day)", 0.0, 6.0, 1.0)
    digital = st.slider("Near-Work Duration (Hrs/Day)", 0.0, 16.0, 6.0)
    work_dist = st.slider("Working Distance (cm)", 10, 60, 35)
    treatment = st.toggle("Enable Clinical Intervention Simulation")

# --- 5. PREDICTION & VISUALIZATION ---
years = np.arange(age_now, 25)
def predict_trajectory(sph, is_treated):
    path, curr = [], sph
    rate = 0.55 if is_treated else 1.0
    for y in years:
        f = np.array([[y, curr, outdoor, digital, parents, int(preterm), int(gd), 8, (1 if gender=="Female" else 0), work_dist, onset]])
        p = model.predict(f)[0]
        curr = curr + ((p - curr) * rate)
        path.append(curr)
    return path

path_untreated = predict_trajectory(od_s, False)
path_treated = predict_trajectory(od_s, True)

st.markdown("---")
v_col, a_col = st.columns([2, 1])

with v_col:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=path_untreated, name='Baseline Progression', line=dict(color='#94A3B8', width=2)))
    if treatment:
        fig.add_trace(go.Scatter(x=years, y=path_treated, name='Intervention Trajectory', line=dict(color='#10B981', width=4)))
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), height=450, yaxis_title="Diopters (D)", xaxis_title="Age")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with a_col:
    st.subheader("Clinical Interventions")
    if work_dist < 30:
        st.error("Recommendation: Increase working distance to >30cm to reduce accommodative lag.")
    if outdoor < 2:
        st.warning("Recommendation: Minimum 120 minutes of outdoor exposure required for retinal health.")
    if treatment:
        st.markdown('<div class="intervention-card"><b>Simulated Impact:</b> Myopia management protocol is projected to stabilize refraction by approximately 45%.</div>', unsafe_allow_html=True)

# --- 6. REPOSITORY & CITATION ---
st.markdown("---")
rep_col1, rep_col2 = st.columns(2)

with rep_col1:
    st.subheader("Data Export & License")
    st.write("This software is provided under the **MIT License**. Data exported is anonymized.")
    metadata = {"age": [age_now], "country": [country], "parents": [parents], "final_rx": [path_treated[-1] if treatment else path_untreated[-1]]}
    csv = pd.DataFrame(metadata).to_csv(index=False).encode('utf-8')
    st.download_button("Export Anonymized CSV", csv, "research_data.csv", "text/csv")
    
    with st.expander("View MIT License Terms"):
        st.code("""
        Copyright (c) 2024 Myopia Insight Contributors
        Permission is hereby granted, free of charge...
        """)

with rep_col2:
    st.subheader("Academic Citation")
    st.write("If using this dataset or tool for research purposes, please cite as follows:")
    st.code("""
@software{myopia_insight_2024,
  author = {Your Name / Organization},
  title = {Myopia Insight AI: Open Data Predictive Engine},
  year = {2024},
  url = {Your Repository URL}
}
    """, language="latex")