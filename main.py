"""
Main Execution Script for NEXUS Brain VHT Pipeline
=================================================
Orchestrates data loading, HBM Fisher-KPP MCMC parameter estimation, 
cohort visualization, future predictive forecasting, and output persistence.
"""

from pathlib import Path
import pandas as pd
import json
import datetime
import pymc
import numpy as np
import pandas as pd
import nibabel as nib

# Import modular components from your repository
from vht_pipeline import run_pipeline
from hbm_Fisher_kpp import build_hierarchical_fisher_kpp_model
from visualization import estimate_and_plot_each_patient
from vht_forecast import run_cohort_forecast  # <--- Forecasting module


def main():
    print("=========================================================")
    print("          NEXUS BRAIN VHT PIPELINE INITIALIZATION        ")
    print("=========================================================")

    # Define hyperparameters for reproducibility tracking
    MCMC_DRAWS = 50       # Increase to 1000+ for final production runs
    MCMC_TUNE = 50        # Increase to 1000+ for final production runs
    TARGET_ACCEPT = 0.90
    RANDOM_SEED = 42

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
        tune_steps=MCMC_TUNE, 
        draw_steps=MCMC_DRAWS
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

    # 4. Save final estimated parameters, growth forecasts, volume history & provenance metadata
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    out_csv = output_dir / "hbm_patient_parameters_summary.csv"
    results_df.to_csv(out_csv, index=False)
    
    forecast_combined = pd.concat(cohort_forecasts.values(), ignore_index=True)
    forecast_csv = output_dir / "vht_tumor_growth_forecasts.csv"
    forecast_combined.to_csv(forecast_csv, index=False)
    
    vol_csv = output_dir / "df_vol.csv"
    df_vol.to_csv(vol_csv, index=False)

    # --- NEW: Build and Save Run Metadata & Provenance JSON ---
    metadata = {
        "execution_timestamp": datetime.datetime.now().isoformat(),
        "cohort_size_patients": int(results_df["Patient_ID"].nunique()),
        "library_versions": {
            "pymc": pymc.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nibabel": nib.__version__
        },
        "mcmc_hyperparameters": {
            "draws": MCMC_DRAWS,
            "tune": MCMC_TUNE,
            "target_accept": TARGET_ACCEPT,
            "random_seed": RANDOM_SEED
        }
    }

    metadata_json_path = output_dir / "run_metadata.json"
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("\n[Step 4/4] Pipeline execution complete!")
    print(f" > Saved posterior parameter summaries to {out_csv}")
    print(f" > Saved tumor growth forecasts to {forecast_csv}")
    print(f" > Saved patient volume history to {vol_csv}")
    print(f" > Saved run metadata & provenance log to {metadata_json_path}")
    print("=========================================================")


if __name__ == "__main__":
    main()