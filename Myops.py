import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- 1. RESEARCH-GRADE UI CONFIGURATION ---
st.set_page_config(page_title="Myopia Insight | Hitha Krishna", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=IBM+Plex+Sans:wght@500;700&display=swap');
    
    /* Professional Canvas */
    .stApp { background-color: #FBFBFE; }
    h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; color: #0F172A; }
    p, span, label { font-family: 'Inter', sans-serif; color: #334155; }

    /* Structural Elements */
    .res-header {
        background-color: #FFFFFF;
        padding: 1.5rem 3rem;
        margin: -6rem -5rem 2rem -5rem;
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .res-title { font-size: 24px; font-weight: 700; color: #0F172A; letter-spacing: -0.5px; }
    .res-tag { background: #059669; padding: 4px 12px; border-radius: 4px; font-size: 11px; color: #FFFFFF; font-weight: 600; text-transform: uppercase; }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Academic Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border: none; color: #64748B; font-weight: 600; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #0F172A; }
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
    
    # Progression heuristics calibrated for 2-23 age range
    prog = ((-0.8 * (24 - age)) * (1 / (age * 0.6)) + (outdoor * 0.2) - (screen * 0.2) - (genetics * 0.4))
    target = current_rx + prog
    
    X = pd.DataFrame({'age': age, 'rx': current_rx, 'out': outdoor, 'scr': screen, 
                      'gen': genetics, 'pre': is_preterm, 'gd': has_gd, 'slp': sleep, 
                      'sex': gender, 'dist': work_dist, 'onset': onset_age})
    return RandomForestRegressor(n_estimators=100, random_state=42).fit(X, target)

model = train_clinical_model()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### Patient Demographics")
    age_now = st.slider("Age at Assessment", 2, 23, 10)
    sex = st.radio("Biological Sex", ["Male", "Female", "Other"], horizontal=True)
    region = st.selectbox("Geographic Region", ["East Asia", "South Asia", "Europe", "North America", "Other"])
    st.divider()
    st.markdown("### Clinical Parameters")
    od_s = st.number_input("OD Sphere (Right)", -15.0, 2.0, -1.50, step=0.25)
    os_s = st.number_input("OS Sphere (Left)", -15.0, 2.0, -1.25, step=0.25)
    onset = st.number_input("Age of Myopia Onset", 2, age_now, 7)
    st.divider()
    st.markdown("### Medical History")
    preterm = st.checkbox("Preterm History")
    maternal_gd = st.checkbox("Maternal Gestational Diabetes")
    parents = st.select_slider("Myopic Parents", options=[0, 1, 2])

# --- 4. NAVIGATION BAR ---
st.markdown(f"""
    <div class="res-header">
        <div class="res-title">Myopia Insight <span style="color:#94A3B8; font-weight:400;">v2.1</span></div>
        <div class="res-tag">Clinical Repository Active</div>
    </div>
""", unsafe_allow_html=True)

# --- 5. MAIN DASHBOARD ---
tab1, tab2 = st.tabs(["Refractive Prediction", "Academic Documentation"])

with tab1:
    # Behavioral Factors Row
    b1, b2, b3 = st.columns(3)
    with b1: outdoor = st.slider("Outdoor Exposure (Hrs/Day)", 0.0, 6.0, 1.0)
    with b2: screen = st.slider("Screen Usage (Hrs/Day)", 0.0, 16.0, 6.0)
    with b3: dist = st.slider("Near-Work Distance (cm)", 10, 60, 35)

    years = np.arange(age_now, 25)
    treatment_active = st.toggle("Simulate Myopia Management Protocol")

    def predict_path(sph, treat):
        path, curr = [], sph
        rate = 0.58 if treat else 1.0
        for y in years:
            f = np.array([[y, curr, outdoor, screen, parents, int(preterm), int(maternal_gd), 8, (1 if sex=="Female" else 0), dist, onset]])
            p = model.predict(f)[0]
            curr = curr + ((p - curr) * rate)
            path.append(curr)
        return path

    p_baseline = predict_path(od_s, False)
    p_treated = predict_path(od_s, True)

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=p_baseline, name='Natural History', line=dict(color='#94A3B8', width=2, dash='dot')))
    if treatment_active:
        fig.add_trace(go.Scatter(x=years, y=p_treated, name='Simulated Treatment', line=dict(color='#059669', width=4)))
    fig.update_layout(template="simple_white", height=450, margin=dict(l=0,r=0,t=40,b=0), yaxis_title="Sphere (D)", xaxis_title="Patient Age")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    final_rx = p_treated[-1] if treatment_active else p_baseline[-1]
    r1.metric("Predicted Refraction @ Age 24", f"{final_rx:.2f} D")
    r2.metric("Refractive Change", f"{final_rx - od_s:.2f} D")

with tab2:
    st.subheader("Repository Information")
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("**License**")
        st.write("Distributed under the MIT License. This software is provided for research purposes. Open Data usage must attribute the author.")
        st.markdown("**Data Contribution**")
        meta = {"age": [age_now], "sex": [sex], "region": [region], "parents": [parents], "final_rx": [final_rx]}
        st.download_button("Download Anonymized Study Metadata", pd.DataFrame(meta).to_csv(index=False).encode('utf-8'), "myopia_metadata.csv", "text/csv")
    
    with col_r:
        st.markdown("**Standard Citation (BibTeX)**")
        st.code(f"""@software{{krishna_myopia_2026,
  author = {{Krishna, Hitha}},
  title = {{Myopia Insight AI: A Refractive Growth Prediction Engine}},
  year = {{2026}},
  version = {{2.1.0}},
  publisher = {{GitHub Repository}},
  url = {{https://github.com/Meerarkrish/Myopia-Insight}},
  note = {{Accessed: {datetime.now().strftime("%Y-%m-%d")}}}}}""", language="latex")