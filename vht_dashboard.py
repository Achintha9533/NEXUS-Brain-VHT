"""
NEXUS Brain VHT: Interactive "What-If" Counterfactual Dashboard
Run with: streamlit run vht_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from vht_forecast import simulate_counterfactual_scenario

st.set_page_config(page_title="NEXUS Brain VHT - Simulation Dashboard", layout="wide")

st.title("🧠 NEXUS Brain VHT: Clinical Counterfactual Simulator")

# Load calibrated patient parameters and volume history
summary_csv = Path("output/hbm_patient_parameters_summary.csv")
vol_csv = Path("output/df_vol.csv")

if not summary_csv.exists() or not vol_csv.exists():
    st.error("Please run main.py first to generate calibrated parameters and volume history.")
    st.stop()

summary_df = pd.read_csv(summary_csv)
vol_df = pd.read_csv(vol_csv)

patient_list = summary_df["Patient_ID"].tolist()

selected_pid = st.sidebar.selectbox("Select Patient", patient_list, key="patient_selector")
p_params = summary_df[summary_df["Patient_ID"] == selected_pid].iloc[0]

# Fetch patient's actual historical data points
p_history = vol_df[vol_df["Patient_ID"] == selected_pid].sort_values("Day")
if not p_history.empty:
    last_row = p_history.iloc[-1]
    last_day = float(last_row["Day"])
    last_volume = float(last_row["Volume_Total"])
else:
    last_day, last_volume = 100.0, 5000.0

st.sidebar.subheader("Calibrated Parameters")
st.sidebar.write(f"**ρ (Proliferation):** {p_params['rho']:.4f}")
st.sidebar.write(f"**D (Diffusion):** {p_params['D']:.4f}")
st.sidebar.write(f"**Wave Speed:** {p_params['wave_speed']:.4f}")
st.sidebar.write(f"**Last Scan Day:** Day {last_day:.0f}")
st.sidebar.write(f"**Last Volume:** {last_volume:,.1f} mm³")

# Extract all calibrated treatment efficacy alphas
alphas = {
    "Chemo": p_params.get("Chemo", 0.0),
    "Rad": p_params.get("Rad", 0.0),
    "Immuno": p_params.get("Immuno", 0.0),
    "Additional1": p_params.get("Additional1", 0.0),
    "Additional2": p_params.get("Additional2", 0.0),
    "Brachy": p_params.get("Brachy", 0.0),
}

st.subheader(f"Scenario Configuration for {selected_pid}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Scenario A: Standard Care (No Therapy)")
    df_a = simulate_counterfactual_scenario(
        patient_id=selected_pid,
        last_day=last_day, last_volume=last_volume,
        rho=p_params['rho'], D=p_params['D'], K=150000,
        calibrated_alphas=alphas,
        treatment_schedule=[],
        forecast_days=180
    )

with col2:
    st.markdown("### Scenario B: Counterfactual Multi-Modal Intervention")
    
    sched_b = []
    
    # 1. Chemotherapy Expander
    with st.expander("💊 Chemotherapy Configuration", expanded=False):
        use_chemo = st.checkbox("Add Chemotherapy", value=False, key="chk_chemo")
        if use_chemo:
            chemo_start = st.slider("Chemo Start Day", int(last_day), int(last_day + 180), int(last_day + 20), key="chemo_start")
            chemo_end = st.slider("Chemo End Day", chemo_start, int(last_day + 180), chemo_start + 60, key="chemo_end")
            sched_b.append({"agent": "Chemo", "start_day": chemo_start, "end_day": chemo_end, "active_flag": 1.0})
        
    # 2. Radiotherapy Expander
    with st.expander("☢️ Radiotherapy Configuration", expanded=False):
        use_rad = st.checkbox("Add Radiotherapy", value=False, key="chk_rad")
        if use_rad:
            rad_start = st.slider("Rad Start Day", int(last_day), int(last_day + 180), int(last_day + 10), key="rad_start")
            rad_end = st.slider("Rad End Day", rad_start, int(last_day + 180), rad_start + 30, key="rad_end")
            sched_b.append({"agent": "Rad", "start_day": rad_start, "end_day": rad_end, "active_flag": 1.0})

    # 3. Immunotherapy Expander
    with st.expander("🛡️ Immunotherapy Configuration", expanded=False):
        use_immuno = st.checkbox("Add Immunotherapy", value=False, key="chk_immuno")
        if use_immuno:
            immuno_start = st.slider("Immuno Start Day", int(last_day), int(last_day + 180), int(last_day + 30), key="immuno_start")
            immuno_end = st.slider("Immuno End Day", immuno_start, int(last_day + 180), immuno_start + 90, key="immuno_end")
            sched_b.append({"agent": "Immuno", "start_day": immuno_start, "end_day": immuno_end, "active_flag": 1.0})

    # 4. Brachytherapy Expander
    with st.expander("⚡ Brachytherapy Configuration", expanded=False):
        use_brachy = st.checkbox("Add Brachytherapy", value=False, key="chk_brachy")
        if use_brachy:
            brachy_day = st.slider("Brachytherapy Day", int(last_day), int(last_day + 180), int(last_day + 15), key="brachy_day")
            sched_b.append({"agent": "Brachy", "start_day": brachy_day, "end_day": brachy_day + 5, "active_flag": 1.0})

    # 5. Surgical Resection Expander
    with st.expander("🔪 Surgical Resection Configuration", expanded=True):
        use_resection = st.checkbox("Perform Surgical Resection", value=True, key="chk_resection")
        if use_resection:
            resect_day = st.slider("Resection Day", int(last_day), int(last_day + 180), int(last_day + 50), key="resect_day")
            resect_pct = st.slider("Resection Extent (%)", 10, 99, 80, key="resect_pct") / 100.0
        else:
            resect_pct = 0.0

    resect_b = [{"day": resect_day, "volume_reduction_fraction": resect_pct}] if use_resection else []

    df_b = simulate_counterfactual_scenario(
        patient_id=selected_pid,
        last_day=last_day, last_volume=last_volume,
        rho=p_params['rho'], D=p_params['D'], K=150000,
        calibrated_alphas=alphas,
        treatment_schedule=sched_b,
        resection_events=resect_b,
        forecast_days=180
    )

# Plot Comprehensive Trajectory (Historical + Forecasts)
st.subheader("📈 Longitudinal & Counterfactual Trajectory Plot")
fig, ax = plt.subplots(figsize=(12, 5))

# Plot historical scans
if not p_history.empty:
    ax.plot(p_history["Day"], p_history["Volume_Total"], 'ko-', linewidth=2, label="Historical Scans")

# Plot forecasts
ax.plot(df_a["Forecast_Day"], df_a["Forecast_Volume"], 'r--', linewidth=2, label="Scenario A: Baseline Growth")
ax.plot(df_b["Forecast_Day"], df_b["Forecast_Volume"], 'g-', linewidth=2, label="Scenario B: Counterfactual Intervention")

ax.set_ylabel("Tumor Volume (mm³)")
ax.set_xlabel("Day from Diagnosis")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
ax.set_title(f"Patient {selected_pid}: Tumor Burden Projections")

st.pyplot(fig)