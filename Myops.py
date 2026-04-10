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
    
    /* Clean Page Background */
    .stApp { background-color: #F8FAFC; }
    
    /* Fixed Header Alignment */
    .header-container {
        background-color: #FFFFFF;
        padding: 2rem;
        border-bottom: 2px solid #F1F5F9;
        margin-bottom: 2rem;
        border-radius: 0 0 15px 15px;
    }
    
    /* Requested Pastel Green Font for Heading */
    .main-title { 
        font-family: 'IBM Plex Sans', sans-serif; 
        font-size: 36px; 
        font-weight: 700; 
        color: #B7E4C7; 
        margin: 0;
    }
    
    .sub-title { 
        font-family: 'Inter', sans-serif; 
        font-size: 14px; 
        color: #64748B; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
    }

    /* Professional Content Cards */
    .clinical-card {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }

    /* Fix for Sidebar Alignment */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
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
    gender = np.random.choice([0, 1], n_samples)
    work_dist = np.random.uniform(15, 60, n_samples)
    onset_age = np.random.uniform(3, 15, n_samples)
    
    prog = ((-0.8 * (24 - age)) * (1 / (age * 0.6)) + (outdoor * 0.2) - (screen * 0.2) - (genetics * 0.4))
    target = current_rx + prog
    X = pd.DataFrame({'age': age, 'rx': current_rx, 'out': outdoor, 'scr': screen, 
                      'gen': genetics, 'pre': is_preterm, 'gd': has_gd, 'slp': 8, 
                      'sex': gender, 'dist': work_dist, 'onset': onset_age})
    return RandomForestRegressor(n_estimators=100, random_state=42).fit(X, target)

model = train_clinical_model()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### Patient Profile")
    age_now = st.slider("Current Age", 2, 23, 10)
    sex = st.radio("Biological Sex", ["Male", "Female", "Other"], horizontal=True)
    region = st.selectbox("Region", ["East Asia", "South Asia", "Europe", "North America", "Other"])
    st.divider()
    st.markdown("### Clinical Data")
    od_s = st.number_input("OD Sphere (Right)", -15.0, 2.0, -1.50, step=0.25)
    os_s = st.number_input("OS Sphere (Left)", -15.0, 2.0, -1.25, step=0.25)
    onset = st.number_input("Onset Age", 2, age_now, 7)
    st.divider()
    st.markdown("### History")
    preterm = st.checkbox("Preterm Birth")
    maternal_gd = st.checkbox("Maternal GD")
    parents = st.select_slider("Myopic Parents", options=[0, 1, 2])

# --- 4. TOP HEADER (ALIGNED) ---
st.markdown("""
    <div class="header-container">
        <div class="main-title">Myopia Insight AI</div>
        <div class="sub-title">Clinical Growth Prediction & Open Data Repository</div>
    </div>
""", unsafe_allow_html=True)

# --- 5. MAIN DASHBOARD ---
col_main, col_data = st.columns([2, 1])

with col_main:
    st.markdown("### Refractive Forecast")
    # Horizontal Inputs
    i1, i2, i3 = st.columns(3)
    with i1: outdoor = st.slider("Outdoor (Hrs/Day)", 0.0, 6.0, 1.0)
    with i2: screen = st.slider("Screen (Hrs/Day)", 0.0, 16.0, 6.0)
    with i3: dist = st.slider("Distance (cm)", 10, 60, 35)

    treatment = st.toggle("Apply Myopia Management Simulation")

    years = np.arange(age_now, 25)
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=p_baseline, name='Natural History', line=dict(color='#94A3B8', width=2, dash='dot')))
    if treatment:
        fig.add_trace(go.Scatter(x=years, y=p_treated, name='Clinical Intervention', line=dict(color='#10B981', width=4)))
    fig.update_layout(template="simple_white", height=400, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_data:
    st.markdown("### Key Metrics")
    final_val = p_treated[-1] if treatment else p_baseline[-1]
    st.metric("Predicted Rx @ Age 24", f"{final_val:.2f} D")
    st.metric("Total Change", f"{final_val - od_s:.2f} D")
    
    st.markdown("---")
    st.markdown("**Intervention Status**")
    if treatment:
        st.success("Treatment Active: Growth velocity reduced.")
    else:
        st.info("Baseline: Natural progression active.")

# --- 6. DOCUMENTATION SECTION (Visible & Aligned) ---
st.markdown("---")
st.markdown("### Research Documentation")
doc_1, doc_2 = st.columns(2)

with doc_1:
    st.markdown("**Legal & Licensing**")
    st.write("This tool and its resulting data are licensed under the **MIT License**. All exported data is fully anonymized to maintain patient privacy.")
    meta_df = pd.DataFrame({"age": [age_now], "parents": [parents], "final_rx": [final_val]})
    st.download_button("Export Anonymized CSV", meta_df.to_csv(index=False).encode('utf-8'), "myopia_data.csv", "text/csv")

with doc_2:
    st.markdown("**Standard Citation (BibTeX)**")
    # Updated to your official GitHub and Name
    st.code(f"""@software{{krishna_myopia_2026,
  author = {{Krishna, Hitha}},
  title = {{Myopia Insight AI: A Refractive Growth Prediction Engine}},
  year = {{2026}},
  version = {{2.1.0}},
  publisher = {{GitHub Repository}},
  url = {{https://github.com/Meerarkrish/Myopia-Insight}},
  note = {{Accessed: {datetime.now().strftime("%Y-%m-%d")}}}}}""", language="latex")