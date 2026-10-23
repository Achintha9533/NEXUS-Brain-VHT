"""
Main Execution Script for NEXUS Brain VHT Pipeline
=================================================
Orchestrates data loading, HBM Fisher-KPP MCMC parameter estimation, 
cohort visualization, future predictive forecasting, and output persistence.
"""

from pathlib import Path
import pandas as pd

# Import modular components from your repository
from vht_pipeline import run_pipeline
from hbm_Fisher_kpp import build_hierarchical_fisher_kpp_model
from visualization import estimate_and_plot_each_patient
from vht_forecast import run_cohort_forecast  # <--- Forecasting module


def main():
    print("=========================================================")
    print("          NEXUS BRAIN VHT PIPELINE INITIALIZATION        ")
    print("=========================================================")

    # 1. Run data ingestion and feature extraction pipeline
    print("\n[Step 1/4] Loading data and building digital patient objects...")
    patient_objects, df_kinetics, df_vol, clinical_df = run_pipeline()

    print(f" > Loaded {len(patient_objects)} patient image folders.")
    print(f" > Extracted kinetics shape: {df_kinetics.shape}")
    print(f" > Extracted volume/exposure shape: {df_vol.shape}")

    if df_vol.empty:
        print("❌ CRITICAL ERROR: Volume dataframe is empty. Check data paths.")
        return

    # 2. Execute HBM MCMC sampling & patient trajectory evaluation
    print("\n[Step 2/4] Executing Hierarchical Bayesian Model (HBM) estimation...")
    results_df = estimate_and_plot_each_patient(
        patient_objects=patient_objects, 
        tune_steps=50,  # Increase to 1000+ for final production runs
        draw_steps=50   # Increase to 1000+ for final production runs
    )

    if results_df.empty:
        print("❌ CRITICAL ERROR: Parameter estimation results summary is empty.")
        return

    # 3. Execute Predictive Forecasting Layer
    print("\n[Step 3/4] Running future tumor growth forecasting engine...")
    cohort_forecasts = run_cohort_forecast(
        results_summary_df=results_df, 
        df_vol=df_vol, 
        forecast_days=180
    )

    # 4. Save final estimated parameters, growth forecasts, & volume history to disk
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    out_csv = output_dir / "hbm_patient_parameters_summary.csv"
    results_df.to_csv(out_csv, index=False)
    
    forecast_combined = pd.concat(cohort_forecasts.values(), ignore_index=True)
    forecast_csv = output_dir / "vht_tumor_growth_forecasts.csv"
    forecast_combined.to_csv(forecast_csv, index=False)
    
    vol_csv = output_dir / "df_vol.csv"
    df_vol.to_csv(vol_csv, index=False)
    
    print("\n[Step 4/4] Pipeline execution complete!")
    print(f" > Saved posterior parameter summaries to {out_csv}")
    print(f" > Saved tumor growth forecasts to {forecast_csv}")
    print(f" > Saved patient volume history to {vol_csv}")
    print("=========================================================")


if __name__ == "__main__":
    main()