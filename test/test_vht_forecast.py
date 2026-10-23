"""
Test Script for NEXUS Brain VHT Forecasting & Counterfactual Module
===================================================================
Verifies correct execution of baseline trajectories, dynamic counterfactual 
schedules, surgical resections, and cohort-wide batch forecasts.
"""

from pathlib import Path
import sys
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vht_forecast import (
    forecast_patient_trajectory,
    simulate_counterfactual_scenario,
    run_cohort_forecast
)

def test_module():
    print("=========================================================")
    print("         NEXUS VHT FORECAST MODULE TEST SUITE            ")
    print("=========================================================")

    # 1. Check if output data files exist
    summary_csv = Path("output/hbm_patient_parameters_summary.csv")
    vol_csv = Path("output/df_vol.csv")

    if not summary_csv.exists() or not vol_csv.exists():
        print("❌ Error: Output CSV files not found. Please run 'main.py' first.")
        return

    summary_df = pd.read_csv(summary_csv)
    vol_df = pd.read_csv(vol_csv)
    
    # Pick the first patient for unit testing
    sample_row = summary_df.iloc[0]
    pid = sample_row["Patient_ID"]
    rho = sample_row["rho"]
    D = sample_row["D"]
    
    # Get last scan history for this patient
    p_history = vol_df[vol_df["Patient_ID"] == pid].sort_values("Day")
    last_row = p_history.iloc[-1]
    last_day = float(last_row["Day"])
    last_volume = float(last_row["Volume_Total"])

    print(f"\n[Test 1] Testing single patient baseline trajectory for {pid}...")
    print(f" > Last Day: {last_day} | Last Volume: {last_volume:.1f} mm³ | ρ: {rho:.4f} | D: {D:.4f}")

    dummy_treatment_params = {"Chemo": sample_row.get("Chemo", 0.1), "Chemo_active_flag": 1.0}
    df_base = forecast_patient_trajectory(
        patient_id=pid,
        last_day=last_day,
        last_volume=last_volume,
        rho=rho,
        D=D,
        K=150000.0,
        treatment_params=dummy_treatment_params,
        forecast_days=180
    )
    print(f" ✅ Baseline forecast generated successfully! Shape: {df_base.shape}")
    print(f"    Final projected volume at Day {df_base.iloc[-1]['Forecast_Day']:.0f}: {df_base.iloc[-1]['Forecast_Volume']:,.1f} mm³")

    print(f"\n[Test 2] Testing counterfactual scenario with Chemotherapy & Resection...")
    calibrated_alphas = {
        "Chemo": sample_row.get("Chemo", 0.1),
        "Rad": sample_row.get("Rad", 0.1)
    }
    treatment_schedule = [
        {"agent": "Chemo", "start_day": last_day + 20, "end_day": last_day + 80, "active_flag": 1.0}
    ]
    resection_events = [
        {"day": last_day + 40, "volume_reduction_fraction": 0.75}  # 75% resection at Day +40
    ]

    df_counter = simulate_counterfactual_scenario(
        patient_id=pid,
        last_day=last_day,
        last_volume=last_volume,
        rho=rho,
        D=D,
        K=150000.0,
        calibrated_alphas=calibrated_alphas,
        treatment_schedule=treatment_schedule,
        resection_events=resection_events,
        forecast_days=180
    )
    print(f" ✅ Counterfactual simulation generated successfully! Shape: {df_counter.shape}")
    print(f"    Volume after resection & treatment at Day {df_counter.iloc[-1]['Forecast_Day']:.0f}: {df_counter.iloc[-1]['Forecast_Volume']:,.1f} mm³")

    print(f"\n[Test 3] Testing cohort-wide batch forecasting engine...")
    cohort_results = run_cohort_forecast(
        results_summary_df=summary_df.head(5),  # Test first 5 patients for speed
        df_vol=vol_df,
        forecast_days=180
    )
    print(f" ✅ Cohort batch forecast successfully generated for {len(cohort_results)} patients.")
    
    print("\n=========================================================")
    print("          ALL MODULE TESTS PASSED SUCCESSFULLY!          ")
    print("=========================================================")

if __name__ == "__main__":
    test_module()