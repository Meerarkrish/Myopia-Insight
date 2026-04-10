import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- 1. INTERFACE STYLING ---
st.set_page_config(page_title="Myopia Insight AI", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500&display=swap');
    .stApp { background-color: #F1F5F9; } 
    .header-box {
        background-color: #FFFFFF; padding: 50px;
        margin: -6rem -5rem 2rem -5rem; width: 110%; 
        border-bottom: 3px solid #B7E4C7;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .headline-main { 
        font-family: 'Poppins', sans-serif; font-size: 42px; 
        font-weight: 700; color: #74C69D !important; 
    }
    .stat-card {
        background: #FFFFFF; padding: 25px; border-radius: 15px; 
        border: 1px solid #E2E8F0; color: #1E293B; margin-bottom: 20px;
    }
    .recommendation-card {
        background: #F0FDF4; border-left: 5px solid #22C55E;
        padding: 15px; margin-top: 10px; border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. HEAVY ML ENGINE ---
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

    onset_impact = np.where(onset_age < 7, 1.4, 1.0)
    dist_impact = np.where(work_dist < 30, (30 - work_dist) * 0.05, 0)
    
    progression = (
        (-0.85 * (24 - age) * onset_impact) * (1 / (age * 0.6)) 
        + (outdoor * 0.22) - (screen * 0.22) - (genetics * 0.40)
        - (is_preterm * 0.95) - (has_gd * 0.50) - dist_impact
    )
    target_rx = current_rx + progression
    
    X = pd.DataFrame({
        'age': age, 'rx': current_rx, 'out': outdoor, 'scr': screen, 
        'gen': genetics, 'pre': is_preterm, 'gd': has_gd, 'slp': sleep, 
        'sex': gender, 'dist': work_dist, 'onset': onset_age
    })
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, target_rx)
    return model

ml_engine = train_clinical_model()

# --- 3. HEADER ---
st.markdown("""
    <div class="header-box">
        <div class="headline-main">Myopia Insight</div>
        <p style="color:#64748B; font-size:18px;">Global Clinical Dataset & Predictive Management (Ages 2–23)</p>
    </div>
""", unsafe_allow_html=True)

# --- 4. INPUTS ---
col_rx, col_hist, col_life = st.columns(3)

with col_rx:
    st.markdown("### 👁️ Patient Profile")
    age_now = st.slider("Current Age", 2, 23, 10)
    gender_input = st.selectbox("Gender", ["Male", "Female", "Other"])
    country = st.selectbox("Country", ["Singapore", "China", "India", "USA", "UK", "Australia", "Other"])
    onset_input = st.number_input("Age at First Glasses", 2, age_now, 7)
    
    st.write("**Right Eye (OD)**")
    od_sph = st.number_input("OD Sphere", -15.0, 2.0, -1.50)
    od_cyl = st.number_input("OD Cylinder", -8.0, 0.0, -0.50)
    
    st.write("**Left Eye (OS)**")
    os_sph = st.number_input("OS Sphere", -15.0, 2.0, -1.25)
    os_cyl = st.number_input("OS Cylinder", -8.0, 0.0, -0.50)

with col_hist:
    st.markdown("### 🏥 Medical History")
    is_preterm = st.toggle("Preterm Birth")
    has_gd = st.toggle("Maternal Gestational Diabetes")
    parents = st.radio("Myopic Parents", [0, 1, 2], horizontal=True)
    heritage = st.toggle("East/South Asian Heritage?")

with col_life:
    st.markdown("### 📱 Daily Lifestyle")
    sunlight = st.slider("Outdoor Light (Hrs/Day)", 0.0, 6.0, 1.0)
    digital = st.slider("Screen Time (Hrs/Day)", 0.0, 16.0, 6.0)
    sleep_hrs = st.slider("Sleep Duration (Hrs/Night)", 4.0, 14.0, 9.0)
    reading_dist = st.slider("Working Distance (cm)", 10, 60, 35)

st.markdown("---")
treatment_col1, treatment_col2 = st.columns([1, 2])
with treatment_col1:
    st.markdown("### 🩺 Myopia Management")
    apply_treatment = st.toggle("Apply Control Treatment")
    treatment_type = st.selectbox("Protocol", ["Optical Defocus", "Atropine", "Combined"], disabled=not apply_treatment)

# --- 5. PREDICTION LOGIC ---
years = np.arange(age_now, 25)
sex_val = 1 if gender_input == "Female" else 0
pre_val, gd_val = (1 if is_preterm else 0), (1 if has_gd else 0)

def get_trajectory(sph, treat=False):
    path_sph, curr_s = [], sph
    eff_rate = 0.52 if treat else 1.0 
    for y in years:
        feats = np.array([[y, curr_s, sunlight, digital, parents, pre_val, gd_val, sleep_hrs, sex_val, reading_dist, onset_input]])
        p_s = ml_engine.predict(feats)[0]
        delta = p_s - curr_s
        p_s = curr_s + (delta * eff_rate)
        if country in ["Singapore", "China"]: p_s -= 0.05 
        path_sph.append(p_s)
        curr_s = p_s
    return path_sph

od_untreated = get_trajectory(od_sph, treat=False)
od_treated = get_trajectory(od_sph, treat=True)
os_untreated = get_trajectory(os_sph, treat=False)
os_treated = get_trajectory(os_sph, treat=True)

# --- 6. VISUALIZATION & ADVICE ---
col_viz, col_advice = st.columns([2, 1])
with col_viz:
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=od_untreated, name='OD Untreated', line=dict(color='#94A3B8', width=2, dash='dot')))
    if apply_treatment:
        fig.add_trace(go.Scatter(x=years, y=od_treated, name='OD with Treatment', line=dict(color='#22C55E', width=5)))
        fig.add_trace(go.Scatter(x=years, y=os_treated, name='OS with Treatment', line=dict(color='#16A34A', width=5, dash='dash')))
    else:
        fig.add_trace(go.Scatter(x=years, y=od_untreated, name='OD Sphere', line=dict(color='#74C69D', width=4)))
    fig.update_layout(title="Predicted Refractive Trajectory", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_advice:
    st.markdown("### 📋 Required Interventions")
    if reading_dist < 30: st.error("📏 **Distance Alert:** Maintain >30cm.")
    if sunlight < 2.5: st.warning("☀️ **More Sunlight:** Aim for 120+ mins.")
    if apply_treatment: st.success(f"**Treatment Benefit:** Saving ~{abs(od_treated[-1] - od_untreated[-1]):.2f}D.")

# --- 7. OPEN DATA EXPORT (CSV) ---
st.markdown("---")
st.markdown("### 🔓 Open Data Contribution")
st.write("Download your anonymized clinical parameters to contribute to global myopia research.")

# Prepare the metadata dictionary
metadata = {
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "age_at_assessment": [age_now],
    "gender": [gender_input],
    "country": [country],
    "onset_age": [onset_input],
    "od_sphere_start": [od_sph],
    "od_cylinder_start": [od_cyl],
    "os_sphere_start": [os_sph],
    "os_cylinder_start": [os_cyl],
    "preterm_birth": [is_preterm],
    "maternal_gd": [has_gd],
    "myopic_parents": [parents],
    "heritage_asian": [heritage],
    "outdoor_hrs": [sunlight],
    "digital_hrs": [digital],
    "sleep_hrs": [sleep_hrs],
    "work_distance_cm": [reading_dist],
    "treatment_applied": [apply_treatment],
    "predicted_final_od": [od_treated[-1] if apply_treatment else od_untreated[-1]],
}

df_metadata = pd.DataFrame(metadata)

# CSV Download Button
csv = df_metadata.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Metadata as CSV (Open Data)",
    data=csv,
    file_name=f"myopia_insight_metadata_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

# --- 8. FOOTER ---
c1, c2, c3 = st.columns(3)
final_val = od_treated[-1] if apply_treatment else od_untreated[-1]
with c1: st.metric("Final Forecast", f"{final_val:.2f}D")
with c2: st.metric("Total Change", f"{final_val - od_sph:.2f}D")
with c3: st.subheader(f"Risk: {'🔴 High' if final_val < -6 else '🟢 Managed'}")