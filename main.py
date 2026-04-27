import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from aux_1 import *

# -------------------------------------------------------------------
# STREAMLIT UI LAYOUT
# -------------------------------------------------------------------
st.title("✈️ Predictive Maintenance: Anomaly & RUL Monitor")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")

# (Load the raw data files)
engine_002_raw = pd.read_csv('data/train2_sample_engine.csv') # Save this from your notebook
engine_004_raw = pd.read_csv('data/test4_sample_engine.csv') # Save this from your notebook

dataset_choice = st.sidebar.radio("Select Engine Stream", ["Standard Engine (FD002)", "Novel Fault Mode (FD004)"])
live_data_stream = engine_002_raw if dataset_choice == "Standard Engine (FD002)" else engine_004_raw

max_cycles = int(live_data_stream['cycle'].max())
current_cycle = st.sidebar.slider("Current Time Cycle", 1, max_cycles, 1, 1)

# --- THE LIVE PIPELINE EXECUTION ---
# 1. Slice data up to the current moment
live_data = live_data_stream[live_data_stream['cycle'] <= current_cycle]

# 2. Run feature engineering
live_df_engineered, final_features = engineer_features_live(live_data)

# 3. Run the Anomaly Monitor
live_df_with_confidence = get_live_confidence(live_df_engineered, final_features)

# 4. Get the latest data point for prediction
latest_point = live_df_with_confidence.iloc[-1]

# --- Dashboard Panels ---
# -------------------------------------------------------------------
# RUL Prediction Panel
# -------------------------------------------------------------------
st.header("📈 RUL Prediction & Risk Analysis")
col1, col2, col3 = st.columns(3)

# 5. Make the final XGBoost prediction
predicted_rul = xgb_model.predict(latest_point[final_features].to_frame().T)[0]
col1.metric("Predicted RUL (Cycles)", f"{predicted_rul:.0f}")

uncertainty_buffer = 15 * (latest_point.get('cycle', 0) / max_cycles)
lower_bound = max(0, predicted_rul - uncertainty_buffer)
col2.metric("Max Safe Cycles", f"{lower_bound:.0f}")

safety_window = 30
prob_failure = 1 / (1 + np.exp((predicted_rul - safety_window) / 5)) 
col3.metric(f"P(Fail < {safety_window}c)", f"{prob_failure:.1%}")


# -------------------------------------------------------------------
# DIAGNOSTIC TRACE MONITOR
# -------------------------------------------------------------------
SENSOR_DICT = {
    'sensor_1': 'Fan Inlet Temp',
    'sensor_2': 'LPC Outlet Temp',
    'sensor_3': 'HPC Outlet Temp',
    'sensor_4': 'LPT Outlet Temp',
    'sensor_5': 'Fan Inlet Pressure',
    'sensor_6': 'Bypass-duct Pressure',
    'sensor_7': 'HPC Outlet Pressure',
    'sensor_8': 'Physical Fan Speed',
    'sensor_9': 'Physical Core Speed',
    'sensor_10': 'Engine pressure ratio',
    'sensor_11': 'HPC Static Pressure',
    'sensor_12': 'Fuel Flow Ratio',
    'sensor_13': 'Corrected Fan Speed',
    'sensor_14': 'Corrected Core Speed',
    'sensor_15': 'Bypass Ratio',
    'sensor_16': 'Burner Fuel-air ratio',
    'sensor_17': 'Bleed Enthalpy',
    'sensor_18': 'Demanded Fan Speed',
    'sensor_19': 'Demanded Corrected Fan Speed',
    'sensor_20': 'HPT Coolant Bleed',
    'sensor_21': 'LPT Coolant Bleed'
}

st.markdown("---")
st.header("⚙️ Physical Diagnostic Monitor")
st.write("Live telemetry for primary degradation indicators (Smoothed via EWMA)")

diag_col1, diag_col2, diag_col3 = st.columns(3)

# Get the latest EWMA-smoothed, normalized values
hpc_pressure = latest_point.get('sensor_11_norm_ewma_mean', 0)
lpt_temp = latest_point.get('sensor_4_norm_ewma_mean', 0)
bypass_ratio = latest_point.get('sensor_15_norm_ewma_mean', 0) # Changed to sensor_15

# Display Metric 1: HPC Static Pressure
with diag_col1:
    st.metric(label=f"1. {SENSOR_DICT['sensor_11']}", value=f"{hpc_pressure:.2f} σ")
    st.markdown(get_status_pill(hpc_pressure), unsafe_allow_html=True)

# Display Metric 2: LPT Outlet Temp
with diag_col2:
    st.metric(label=f"2. {SENSOR_DICT['sensor_4']}", value=f"{lpt_temp:.2f} σ")
    st.markdown(get_status_pill(lpt_temp), unsafe_allow_html=True)

# Display Metric 3: Bypass Ratio
with diag_col3:
    st.metric(label=f"3. {SENSOR_DICT['sensor_15']}", value=f"{bypass_ratio:.2f} σ")
    st.markdown(get_status_pill(bypass_ratio), unsafe_allow_html=True)

# Expandable Glossary for the mechanics
with st.expander("📚 View Full Engine Sensor Glossary"):
    glossary_df = pd.DataFrame(list(SENSOR_DICT.items()), columns=['Sensor ID', 'Physical Component'])
    st.table(glossary_df)


# -------------------------------------------------------------------
# Unsupervised Anomaly Monitor
# -------------------------------------------------------------------
st.markdown("---")

st.header("📉 Unsupervised Health & Anomaly Monitor")
col_pca, col_gauge = st.columns([2, 1])

# 6. Get the live confidence score
confidence_score = latest_point['Confidence_Score']

with col_gauge:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score,
        domain={'x': [0, 1], 'y':[0, 1]},
        title={'text': "Model Confidence Score"},
        gauge={'axis': {'range':[0, 100]},
               'bar': {'color': "darkblue"},
               'steps':[
                   {'range': [0, 50], 'color': "red"},
                   {'range':[50, 85], 'color': "yellow"},
                   {'range': [85, 100], 'color': "green"}],
               }))
    gauge.update_layout(height=300)
    st.plotly_chart(gauge, use_container_width=True)

with col_pca:
    pca_fig = go.Figure()
    # 1. DRAW THE HEALTHY ZONE (The GMM Boundary)
    pca_fig.add_shape(type="rect",
        xref="x", yref="y",
        x0=-12, y0=-7, x1=4, y1=6, # <--- Updated to match your new PCA plot's dark green zone
        fillcolor="lightgreen", opacity=0.15, line_color="green", line_width=1
    )
    pca_fig.add_annotation(x=-5, y=6, text="✅ GMM Healthy Manifold", showarrow=False, font=dict(color="lightgreen"))

    # 2. Plot the Engine History (Grey line)
    pca_fig.add_trace(go.Scatter(
        x=live_data['PC1'], y=live_data['PC2'],
        mode='lines', line=dict(color='lightgray', width=2), name='Engine History'
    ))

    # 3. Plot the Current State (Red X)
    pca_fig.add_trace(go.Scatter(
        x=[latest_point['PC1']], y=[latest_point['PC2']],
        mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Current State'
    ))

    pca_fig.update_layout(
        title="Live Engine Trajectory on the PCA Health Manifold",
        xaxis_title="Principal Component 1", 
        yaxis_title="Principal Component 2",
        xaxis=dict(range=[-20, 25]), # Widened to see the full FD004 spread
        yaxis=dict(range=[-20, 25])  # Widened to catch the extreme drops/spikes!
    )

    st.plotly_chart(pca_fig, use_container_width=True)
