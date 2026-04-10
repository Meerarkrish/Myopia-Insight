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
    
    /* Reverted to Professional Light Background */
    .stApp { background-color: #F8FAFC; }
    
    /* Header Styling */
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
        color: #B7E4C7; /* Requested Pastel Green */
        margin: 0;
    }
    
    .sub-title { font-family: 'Inter', sans-serif; font-size: 13px; color: #94A3B8; letter-spacing: 1px; }

    /* Custom Neon/Fluorescent Sliders */
    div[data-baseweb="slider"] > div { background: #22D3EE !important; } /* Fluorescent Cyan Track */
    div[role="slider"] { background-color: #22D3EE !important; border: 2px solid #FFFFFF !important; }
    
    /* Recommendation Styles */
    .rec-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid;
        font-family: 'Inter', sans-serif;
    }
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
st.markdown("""
    <div class="header-container">
        <div class="main-title">Myopia Insight AI</div>
        <div class="sub-title">Clinical Decision Support • Repository: Meerarkrish/Myopia-Insight</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR (Restored All Options) ---
with st.sidebar:
    st.header("Patient Profile")
    age_now = st.slider("Current Age", 2, 23, 10)
    country = st.selectbox("Country/Region", ["Singapore", "China", "India", "USA", "UK", "Australia", "Other"])
    heritage = st.toggle("East/South Asian Heritage")
    st.divider()
    
    st.subheader("OD (Right Eye)")
    od_sph = st.number_input("OD Sphere (D)", -15.0, 2.0, -1.50, step=0.25)
    od_cyl = st.number_input("OD Cylinder (D)", -8.0, 0.0, -0.50, step=0.25)
    
    st.subheader("OS (Left Eye)")
    os_sph = st.number_input("OS Sphere (D)", -15.0, 2.0, -1.25, step=0.25)
    os_cyl = st.number_input("OS Cylinder (D)", -8.0, 0.0, -0.50, step=0.25)
    st.divider()
    
    st.subheader("Medical History")
    onset = st.number_input("Age of Myopia Onset", 2, age_now, 7)
    parents = st.radio("Myopic Parents", [0, 1, 2], horizontal=True)
    is_preterm = st.toggle("Preterm Birth History")
    has_gd = st.toggle("Maternal Gestational Diabetes")

# --- 5. MAIN DASHBOARD ---
c1, c2, c3 = st.columns(3)
with c1: outdoor = st.slider("Outdoor Time (Hrs/Day)", 0.0, 6.0, 1.0)
with c2: screen = st.slider("Screen/Near-Work (Hrs/Day)", 0.0, 16.0, 6.0)
with c3: dist = st.slider("Working Distance (cm)", 10, 60, 35)

treatment = st.toggle("Enable Clinical Intervention Simulation")

# Prediction Logic
years = np.arange(age_now, 25)
def get_path(sph, treat):
    path, curr = [], sph
    rate = 0.55 if treat else 1.0
    for y in years:
        f = np.array([[y, curr, outdoor, screen, parents, int(is_preterm), int(has_gd), 8, 1, dist, onset]])
        p = model.predict(f)[0]
        curr = curr + ((p - curr) * rate)
        path.append(curr)
    return path

p_baseline = get_path(od_sph, False)
p_treated = get_path(od_sph, True)

# Layout: Chart vs Recommendations
chart_col, rec_col = st.columns([2, 1])

with chart_col:
    fig = go.Figure()
    # Baseline: Neon Cyan Dash
    fig.add_trace(go.Scatter(x=years, y=p_baseline, name='Natural History', line=dict(color='#22D3EE', width=2, dash='dot')))
    if treatment:
        # Treatment: Fluorescent Purple Solid
        fig.add_trace(go.Scatter(x=years, y=p_treated, name='Intervention Path', line=dict(color='#A855F7', width=5)))
    fig.update_layout(template="simple_white", height=450, yaxis_title="Sphere (D)", xaxis_title="Age (Years)")
    st.plotly_chart(fig, use_container_width=True)

with rec_col:
    st.subheader("Lifestyle Recommendations")
    
    if dist < 30:
        st.markdown(f'<div class="rec-card rec-warning"><b>Proximity Alert:</b> Distance is {dist}cm. Aim for >30cm.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rec-card rec-success"><b>Optimal Distance:</b> Healthy working habits detected.</div>', unsafe_allow_html=True)
        
    if outdoor < 2:
        st.markdown('<div class="rec-card rec-warning"><b>Sunlight Deficiency:</b> Minimum 120 mins/day recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rec-card rec-success"><b>Sunlight Optimal:</b> Retinal dopamine stimulus met.</div>', unsafe_allow_html=True)

    st.divider()
    final_val = p_treated[-1] if treatment else p_baseline[-1]
    st.metric("Predicted Rx @ Age 24", f"{final_val:.2f} D")

# --- 6. REPOSITORY & DOCUMENTATION ---
st.divider()
doc_a, doc_b = st.columns(2)

with doc_a:
    st.markdown("**Academic Citation (BibTeX)**")
    st.code(f"""@software{{krishna_myopia_2026,
  author = {{Krishna, Hitha}},
  title = {{Myopia Insight AI: A Refractive Growth Prediction Engine}},
  year = {{2026}},
  url = {{https://github.com/Meerarkrish/Myopia-Insight}}
}}""", language="latex")
    
    meta = {"timestamp": [datetime.now()], "country": [country], "final_rx": [final_val]}
    st.download_button("Download Research CSV", pd.DataFrame(meta).to_csv(index=False).encode('utf-8'), "myopia_data.csv")

with doc_b:
    st.markdown("**License & Repository**")
    st.write(f"Source Code: [GitHub Link](https://github.com/Meerarkrish/Myopia-Insight)")
    with st.expander("View MIT License Terms"):
        st.text(f"Copyright (c) {datetime.now().year} Hitha Krishna\n\nPermission is hereby granted, free of charge...")