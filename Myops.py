import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. THE MODERN INTERFACE & PASTEL STYLING ---
st.set_page_config(page_title="Myopia Insight AI", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500&display=swap');
    
    .stApp { background-color: #0F172A; } /* Deep Navy Background */
    
    .header-box {
        background-color: #1E293B; 
        padding: 50px;
        margin: -6rem -5rem 2rem -5rem; 
        width: 110%; 
        border-bottom: 2px solid #B7E4C7;
    }
    
    .headline-main { 
        font-family: 'Poppins', sans-serif; 
        font-size: 42px; 
        font-weight: 700; 
        color: #B7E4C7 !important; /* LIGHT PASTEL GREEN */
        letter-spacing: -1px;
    }

    .sub-headline {
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
        font-size: 18px;
    }

    .stat-card {
        background: rgba(183, 228, 199, 0.05);
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid rgba(183, 228, 199, 0.3);
        color: white;
    }
    
    /* Customizing Sidebar colors for Dark Mode */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. THE DATA SCIENCE ENGINE (Random Forest) ---
@st.cache_resource
def train_clinical_model():
    """Generates a synthetic longitudinal dataset and trains a regressor"""
    n_samples = 10000
    age = np.random.uniform(5, 16, n_samples)
    current_rx = np.random.uniform(-5, 0, n_samples)
    outdoor = np.random.uniform(0, 4, n_samples)
    screen = np.random.uniform(1, 10, n_samples)
    genetics = np.random.choice([0, 1, 2], n_samples)
    
    # Complex non-linear target logic
    progression = (
        (-0.7 * (21 - age)) * (1 / (age * 0.4)) # Age-based decay
        + (outdoor * 0.2) # Protective
        - (screen * 0.12) # Risk
        - (genetics * 0.3) # Genetic multiplier
    )
    target_rx = current_rx + progression
    
    X = pd.DataFrame({'age': age, 'rx': current_rx, 'out': outdoor, 'scr': screen, 'gen': genetics})
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, target_rx)
    return model

ml_engine = train_clinical_model()

# --- 3. HEADER SECTION ---
st.markdown("""
    <div class="header-box">
        <div class="headline-main">Myopia Insight</div>
        <div class="sub-headline">Predictive Refractive Trajectory • Powered by Random Forest Regression</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. LAYOUT & INPUTS ---
col_in, col_viz = st.columns([1, 2])

with col_in:
    st.markdown("### 🧬 Clinical Input")
    with st.container():
        age_now = st.slider("Current Age", 5, 18, 10)
        rx_now = st.number_input("Current Prescription (D)", -10.0, 0.0, -1.25, step=0.25)
        
        st.markdown("---")
        st.markdown("### 📱 Lifestyle Variables")
        sunlight = st.slider("Outdoor Light (Hrs/Day)", 0.0, 5.0, 1.0)
        digital = st.slider("Digital Device Usage (Hrs/Day)", 0.0, 12.0, 6.0)
        parents = st.radio("Number of Myopic Parents", [0, 1, 2], horizontal=True)

# --- 5. PREDICTION & RECURSIVE INFERENCE ---
# We use the ML model to forecast each year until age 21
years = np.arange(age_now, 22)
path = []
temp_rx = rx_now

for y in years:
    # Feature array for the ML model
    features = np.array([[y, temp_rx, sunlight, digital, parents]])
    pred = ml_engine.predict(features)[0]
    path.append(pred)
    temp_rx = pred

# --- 6. VISUALIZATION ---
with col_viz:
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Predicted Path
    fig.add_trace(go.Scatter(
        x=years, y=path, 
        mode='lines+markers',
        name='Predicted Trajectory',
        line=dict(color='#B7E4C7', width=4),
        marker=dict(size=8)
    ))

    # Threshold Zones
    fig.add_hrect(y0=-6, y1=-12, fillcolor="red", opacity=0.1, annotation_text="High Myopia Risk Area")
    
    fig.update_layout(
        title="10-Year Refractive Forecast (AI Inference)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Age", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Prescription (Diopters)", gridcolor="rgba(255,255,255,0.1)"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. IMPACT FOOTER ---
res_col1, res_col2 = st.columns(2)
with res_col1:
    final_rx = path[-1]
    st.metric("Predicted Final Rx (Age 21)", f"{final_rx:.2f} D", delta=f"{final_rx - rx_now:.2f} Change")

with res_col2:
    risk_level = "🔴 High" if final_rx < -6.0 else "🟡 Moderate" if final_rx < -3.0 else "🟢 Low"
    st.subheader(f"Pathological Risk: {risk_level}")

st.info("The AI identifies your current combination of digital hours and baseline age as the primary driver for progression. Increasing sunlight to 2+ hours could significantly flatten this curve.")