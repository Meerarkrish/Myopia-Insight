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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=IBM+Plex+Sans:wght@500;700&family=Lexend:wght@700&display=swap');
    
    /* Global Styles */
    .stApp { background-color: #0F172A; } /* Dark Slate Background */
    h1, h2, h3 { font-family: 'Lexend', sans-serif; color: #F1F5F9; }
    p, span, label { font-family: 'IBM Plex Sans', sans-serif; color: #94A3B8; }

    /* Custom Header with Light Pastel Green Title */
    .res-header {
        background-color: #1E293B;
        padding: 1.5rem 3rem;
        margin: -6rem -5rem 2rem -5rem;
        border-bottom: 2px solid #334155;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .res-title { 
        font-family: 'Lexend', sans-serif;
        font-size: 28px; 
        font-weight: 700; 
        color: #B7E4C7; /* Requested Pastel Green Title */
        letter-spacing: -0.5px; 
    }
    .res-tag { background: #334155; padding: 4px 12px; border-radius: 4px; font-size: 11px; color: #94A3B8; font-weight: 600; text-transform: uppercase; }

    /* Professional Content Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Input Styling - Shift sliders from red to fluorescent cyan */
    div[data-baseweb="slider"] > div { background: #22D3EE !important; } /* Fluorescent Cyan track */
    div[role="slider"] { background-color: #22D3EE !important; border: 2px solid #FFFFFF !important; } /* Slider Handle */
    
    /* Input number boxes */
    input[type=number] { background-color: #1E293B !important; color: #22D3EE !important; border: 1px solid #334155 !important; }
    
    /* Toggle Switch from green to fluorescent purple */
    div[data-testid="stCheckbox"] > label > div[role="switch"] > div { background-color: #A855F7 !important; }

    /* Lifestyle Recommendation Cards */
    .rec-card {
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
    .rec-warning { background-color: #2D2010; border-color: #F59E0B; color: #FCD34D; }
    .rec-success { background-color: #102D20; border-color: #10B981; color: #6EE7B7; }
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

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### Patient Demographics")
    age_now = st.slider("Age at Assessment", 2, 23, 10)
    sex = st.radio("Biological Sex", ["Male", "Female", "Other"], horizontal=True)
    region = st.selectbox("Global Region", ["East Asia", "South Asia", "Europe", "North America", "Other"])
    st.divider()
    st.markdown("### Clinical Parameters")
    od_s = st.number_input("OD Sphere (Right)", -15.0, 2.0, -1.50, step=0.25)
    os_s = st.number_input("OS Sphere (Left)", -15.0, 2.0, -1.25, step=0.25)
    onset = st.number_input("Age of Myopia Onset", 2, age_now, 7)
    st.divider()
    st.markdown("### Medical History")
    preterm = st.checkbox("Preterm History")
    maternal_gd = st.checkbox("Maternal GD History")
    parents = st.select_slider("Myopic Parents", options=[0, 1, 2])

# --- 4. NAVIGATION BAR ---
st.markdown(f"""
    <div class="res-header">
        <div class="res-title">Myopia Insight <span style="color:#94A3B8; font-weight:400; font-size:16px;">v2.1</span></div>
        <div class="res-tag">Clinical Repository Active</div>
    </div>
""", unsafe_allow_html=True)

# --- 5. MAIN DASHBOARD ---
# Behavioral Inputs Row
b1, b2, b3 = st.columns(3)
with b1: outdoor = st.slider("Outdoor Exposure (Hrs/Day)", 0.0, 6.0, 1.0)
with b2: screen = st.slider("Screen Usage (Hrs/Day)", 0.0, 16.0, 6.0)
with b3: dist = st.slider("Near-Work Distance (cm)", 10, 60, 35)

years = np.arange(age_now, 25)
# Using Fluorescent Purple for treatment toggle
treatment_active = st.toggle("Simulate Myopia Management Protocol (Atropine/Optical)")

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

# Layout: Chart vs Recommendations
col_chart, col_rec = st.columns([2, 1])

with col_chart:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    fig = go.Figure()
    
    # Baseline: Neon Cyan Dash
    fig.add_trace(go.Scatter(x=years, y=p_baseline, name='Natural History', line=dict(color='#22D3EE', width=2, dash='dot')))
    
    if treatment_active:
        # Treatment: Fluorescent Purple Solid
        fig.add_trace(go.Scatter(x=years, y=p_treated, name='Simulated Treatment', line=dict(color='#A855F7', width=5)))
        
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        height=450, 
        margin=dict(l=0,r=0,t=40,b=0), 
        yaxis_title="Sphere (D)", 
        xaxis_title="Patient Age",
        yaxis=dict(gridcolor='#334155'),
        xaxis=dict(gridcolor='#334155')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_rec:
    st.subheader("Lifestyle Interventions")
    
    if dist < 30:
        st.markdown(f'<div class="rec-card rec-warning"><b>Distance Alert:</b> Working distance is {dist}cm. Increase to >30cm.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rec-card rec-success"><b>Optimal Distance:</b> Maintain >30cm near work.</div>', unsafe_allow_html=True)
        
    if outdoor < 2:
        st.markdown('<div class="rec-card rec-warning"><b>Light Deficiency:</b> Minimum 120 mins of sunlight required.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rec-card rec-success"><b>Sufficient Light:</b> Target 2+ hours met.</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    final_rx = p_treated[-1] if treatment_active else p_baseline[-1]
    # Replaced red metric text with high-tech fluorescent cyan
    r1.metric("Predicted Refraction @ 24", f"{final_rx:.2f}D", delta_color="normal")
    r2.metric("Cumulative Change", f"{final_rx - od_s:.2f}D", delta_color="normal")

# --- 6. REPOSITORY & CITATION ---
st.divider()
doc_a, doc_b = st.columns(2)

with doc_a:
    st.markdown("**Academic Citation (BibTeX)**")
    # Clean high-contrast code block for researchers
    st.code(f"""@software{{krishna_myopia_2026,
  author = {{Krishna, Hitha}},
  title = {{Myopia Insight AI: A Refractive Growth Prediction Engine}},
  year = {{2026}},
  version = {{2.1.0}},
  publisher = {{GitHub Repository}},
  url = {{https://github.com/Meerarkrish/Myopia-Insight}},
  note = {{Accessed: {datetime.now().strftime("%Y-%m-%d")}}}}}""", language="latex")
    
    # Metadata Export
    meta = {"timestamp": [datetime.now()], "sex": [sex], "region": [region], "parents": [parents], "final_rx": [final_rx]}
    st.download_button("Download Anonymized CSV Metadata", pd.DataFrame(meta).to_csv(index=False).encode('utf-8'), "myopia_research.csv", "text/csv")

with doc_b:
    st.markdown("**Documentation & License**")
    st.write(f"Source Code: [Meerarkrish/Myopia-Insight](https://github.com/Meerarkrish/Myopia-Insight)")
    st.info("Licensed under the MIT Open Source License. Public usage requires academic attribution.")