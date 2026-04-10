import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. THE MODERN INTERFACE & LIGHT GREY STYLING ---
st.set_page_config(page_title="Myopia Insight AI", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500&display=swap');
    
    /* Light Grey Background */
    .stApp { background-color: #F1F5F9; } 
    
    .header-box {
        background-color: #FFFFFF; 
        padding: 50px;
        margin: -6rem -5rem 2rem -5rem; 
        width: 110%; 
        border-bottom: 3px solid #B7E4C7; /* Pastel Green Accent */
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    .headline-main { 
        font-family: 'Poppins', sans-serif; 
        font-size: 42px; 
        font-weight: 700; 
        color: #74C69D !important; /* DARKER PASTEL GREEN for readability on white */
        letter-spacing: -1px;
    }

    .sub-headline {
        color: #64748B;
        font-family: 'Inter', sans-serif;
        font-size: 18px;
        margin-top: 5px;
    }

    /* Cards for the content */
    .stat-card {
        background: #FFFFFF;
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #E2E8F0;
        color: #1E293B;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    /* Text color for standard streamlit labels */
    .stSlider label, .stNumberInput label, .stRadio label {
        color: #334155 !important;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. THE DATA SCIENCE ENGINE ---
@st.cache_resource
def train_clinical_model():
    n_samples = 5000
    age = np.random.uniform(5, 16, n_samples)
    current_rx = np.random.uniform(-5, 0, n_samples)
    outdoor = np.random.uniform(0, 4, n_samples)
    screen = np.random.uniform(1, 10, n_samples)
    genetics = np.random.choice([0, 1, 2], n_samples)
    
    progression = (
        (-0.7 * (21 - age)) * (1 / (age * 0.4)) 
        + (outdoor * 0.2) 
        - (screen * 0.12) 
        - (genetics * 0.3) 
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
        <div class="sub-headline">AI-Driven Refractive Trajectory • Predictive Public Health Tool</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. LAYOUT ---
col_in, col_viz = st.columns([1, 2])

with col_in:
    st.markdown("### 🧬 Patient Metrics")
    age_now = st.slider("Current Age", 5, 18, 10)
    rx_now = st.number_input("Current Prescription (D)", -10.0, 0.0, -1.25, step=0.25)
    
    st.markdown("---")
    st.markdown("### 📱 Behavioral Factors")
    sunlight = st.slider("Outdoor Light (Hrs/Day)", 0.0, 5.0, 1.0)
    digital = st.slider("Digital Device Usage (Hrs/Day)", 0.0, 12.0, 6.0)
    parents = st.radio("Number of Myopic Parents", [0, 1, 2], horizontal=True)

# --- 5. PREDICTION ---
years = np.arange(age_now, 22)
path = []
temp_rx = rx_now
for y in years:
    features = np.array([[y, temp_rx, sunlight, digital, parents]])
    pred = ml_engine.predict(features)[0]
    path.append(pred)
    temp_rx = pred

# --- 6. VISUALIZATION ---
with col_viz:
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years, y=path, 
        mode='lines+markers',
        name='Predicted Trajectory',
        line=dict(color='#74C69D', width=4), # Darker pastel green for the line
        marker=dict(size=8, color='#40916C')
    ))

    fig.add_hrect(y0=-6, y1=-12, fillcolor="#FF0000", opacity=0.05, annotation_text="Pathological Myopia Risk")
    
    fig.update_layout(
        title="10-Year Forecast (ML Projection)",
        template="plotly_white", # Changed to white template
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Age", showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="Prescription (Diopters)", showgrid=True, gridcolor="#E2E8F0"),
        height=500,
        font=dict(color="#1E293B")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. RESULTS ---
st.markdown("---")
res1, res2, res3 = st.columns(3)
with res1:
    st.metric("Final Rx (Age 21)", f"{path[-1]:.2f} D")
with res2:
    total_change = path[-1] - rx_now
    st.metric("Total Predicted Change", f"{total_change:.2f} D")
with res3:
    status = "🔴 High Risk" if path[-1] < -6.0 else "🟡 Moderate" if path[-1] < -3.0 else "🟢 Healthy"
    st.subheader(f"Status: {status}")