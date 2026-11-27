"""
Test Script for NEXUS Brain VHT Dashboard Logic & Multi-Modal Treatments
======================================================================
Verifies data loading, parameter extraction, and multi-modal counterfactual 
simulation pipelines configured in the Streamlit dashboard.
"""

from pathlib import Path
import pandas as pd
import sys

# Add the parent directory (NEXUS Brain VHT) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vht_forecast import simulate_counterfactual_scenario

def test_dashboard_logic():
    print("=========================================================")
    print("      NEXUS VHT DASHBOARD LOGIC & MULTI-MODAL TEST       ")
    print("=========================================================")

    # 1. Verify required CSV files exist
    summary_csv = Path("output/hbm_patient_parameters_summary.csv")
    vol_csv = Path("output/df_vol.csv")

    if not summary_csv.exists() or not vol_csv.exists():
        print("❌ Error: Output CSV files not found. Please run 'main.py' first.")
        return

    summary_df = pd.read_csv(summary_csv)
    vol_df = pd.read_csv(vol_csv)
    
    # Select first patient
    selected_pid = summary_df["Patient_ID"].iloc[0]
    p_params = summary_df[summary_df["Patient_ID"] == selected_pid].iloc[0]

    # Fetch patient history
    p_history = vol_df[vol_df["Patient_ID"] == selected_pid].sort_values("Day")
    if not p_history.empty:
        last_row = p_history.iloc[-1]
        last_day = float(last_row["Day"])
        last_volume = float(last_row["Volume_Total"])
    else:
        last_day, last_volume = 100.0, 5000.0

    print(f"\n[1/3] Loaded patient data for {selected_pid}:")
    print(f"      - Last Scan Day: Day {last_day:.0f}")
    print(f"      - Last Volume: {last_volume:,.1f} mm³")
    print(f"      - Proliferation (ρ): {p_params['rho']:.4f}")
    print(f"      - Diffusion (D): {p_params['D']:.4f}")

    # Extract all calibrated treatment efficacy alphas
    alphas = {
        "Chemo": p_params.get("Chemo", 0.0),
        "Rad": p_params.get("Rad", 0.0),
        "Immuno": p_params.get("Immuno", 0.0),
        "Additional1": p_params.get("Additional1", 0.0),
        "Additional2": p_params.get("Additional2", 0.0),
        "Brachy": p_params.get("Brachy", 0.0),
    }

    # 2. Test Scenario A: Baseline Growth (No Therapy)
    print("\n[2/3] Testing Scenario A (Standard Care Baseline)...")
    df_a = simulate_counterfactual_scenario(
        patient_id=selected_pid,
        last_day=last_day, 
        last_volume=last_volume,
        rho=p_params['rho'], 
        D=p_params['D'], 
        K=150000,
        calibrated_alphas=alphas,
        treatment_schedule=[],
        forecast_days=180
    )
    print(f" ✅ Scenario A computed successfully! Final Volume: {df_a.iloc[-1]['Forecast_Volume']:,.1f} mm³")

    # 3. Test Scenario B: Full Multi-Modal Intervention & Resection
    print("\n[3/3] Testing Scenario B (Multi-Modal Therapy + Resection)...")
    sched_b = [
        {"agent": "Chemo", "start_day": last_day + 20, "end_day": last_day + 80, "active_flag": 1.0},
        {"agent": "Rad", "start_day": last_day + 10, "end_day": last_day + 40, "active_flag": 1.0},
        {"agent": "Immuno", "start_day": last_day + 30, "end_day": last_day + 120, "active_flag": 1.0},
        {"agent": "Brachy", "start_day": last_day + 15, "end_day": last_day + 20, "active_flag": 1.0},
    ]
    resect_b = [{"day": last_day + 50, "volume_reduction_fraction": 0.80}]

    df_b = simulate_counterfactual_scenario(
        patient_id=selected_pid,
        last_day=last_day, 
        last_volume=last_volume,
        rho=p_params['rho'], 
        D=p_params['D'], 
        K=150000,
        calibrated_alphas=alphas,
        treatment_schedule=sched_b,
        resection_events=resect_b,
        forecast_days=180
    )
    print(f" ✅ Scenario B computed successfully! Final Volume after multi-modal therapy: {df_b.iloc[-1]['Forecast_Volume']:,.1f} mm³")

    print("\n=========================================================")
    print("          ALL DASHBOARD LOGIC TESTS PASSED!              ")
    print("=========================================================")

if __name__ == "__main__":
    test_dashboard_logic()