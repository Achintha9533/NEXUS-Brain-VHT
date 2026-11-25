"""
Main Execution Script for NEXUS Brain VHT Pipeline
=================================================
Orchestrates data loading, HBM Fisher-KPP MCMC parameter estimation,
cohort visualization, future predictive forecasting, and output persistence.
"""

from pathlib import Path
import datetime
import json
import arviz as az
import nibabel as nib
import numpy as np
import pandas as pd
import pymc as pm

# Import modular components from your repository
from vht_pipeline import run_pipeline
from hbm_Fisher_kpp import build_hierarchical_fisher_kpp_model  # noqa: F401
import visualization
from visualization import estimate_and_plot_each_patient
from vht_forecast import run_cohort_forecast


HYPERPRIOR_VARS = [
    "mu_log_prod",
    "mu_log_ratio",
    "sigma_log_prod",
    "sigma_log_ratio",
    "mu_log_rho",
    "mu_log_D",
    "mu_log_K",
    "sigma_log_K",
    "mu_alpha_chemo",
    "mu_alpha_rad",
    "mu_alpha_immuno",
    "mu_alpha_add1",
    "mu_alpha_add2",
    "mu_alpha_brachy",
    "sigma_obs_log_v",
    "sigma_obs_log_inv",
]


def _extract_trace_diagnostics(results_df: pd.DataFrame) -> dict:
    """Extracts real MCMC convergence metrics if an InferenceData trace is available."""
    trace = getattr(visualization, "latest_trace", None) or getattr(
        visualization, "trace", None
    )
    if trace is None and hasattr(results_df, "attrs"):
        trace = results_df.attrs.get("trace") or results_df.attrs.get("idata")

    if trace is None:
        return {
            "status": "unverified_no_trace_exposed",
            "r_hat_threshold": "<= 1.01",
            "max_r_hat": None,
            "min_ess_bulk": None,
            "min_ess_tail": None,
            "divergent_transitions": None,
        }

    available_vars = [
        v for v in HYPERPRIOR_VARS if v in trace.posterior.data_vars
    ]
    summary = az.summary(
        trace,
        var_names=available_vars if available_vars else None,
    )

    max_r_hat = float(summary["r_hat"].max()) if "r_hat" in summary else None
    min_ess_bulk = (
        float(summary["ess_bulk"].min()) if "ess_bulk" in summary else None
    )
    min_ess_tail = (
        float(summary["ess_tail"].min()) if "ess_tail" in summary else None
    )

    divergences = None
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        divergences = int(np.asarray(trace.sample_stats["diverging"].values).sum())

    passed = (
        max_r_hat is not None
        and max_r_hat <= 1.01
        and (divergences is None or divergences == 0)
    )

    return {
        "status": "verified" if passed else "needs_review",
        "r_hat_threshold": "<= 1.01",
        "max_r_hat": max_r_hat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergent_transitions": divergences,
    }


def main():
    print("=========================================================")
    print("          NEXUS BRAIN VHT PIPELINE INITIALIZATION        ")
    print("=========================================================")

    # Define hyperparameters for reproducibility tracking
    MCMC_DRAWS = 1000
    MCMC_TUNE = 2000
    TARGET_ACCEPT = 0.99
    RANDOM_SEED = 42
    FORECAST_DAYS = 180

    # 1. Run data ingestion and feature extraction pipeline
    print("\n[Step 1/4] Loading data and building digital patient objects...")
    patient_objects, df_kinetics, df_vol, clinical_df = run_pipeline()

    print(f" > Loaded {len(patient_objects)} patient image folders.")
    print(f" > Extracted kinetics shape: {df_kinetics.shape}")
    print(f" > Extracted volume/exposure shape: {df_vol.shape}")

    if df_vol.empty:
        print("❌ CRITICAL ERROR: Volume dataframe is empty. Check data paths.")
        return

    has_invasive_col = "Volume_Invasive" in df_vol.columns
    print(
        " > Invasive volume observable ('Volume_Invasive') in df_vol: "
        f"{has_invasive_col}"
    )

    # 2. Execute HBM MCMC sampling & patient trajectory evaluation
    print("\n[Step 2/4] Executing Hierarchical Bayesian Model (HBM) estimation...")
    results_df = estimate_and_plot_each_patient(
        patient_objects=patient_objects,
        tune_steps=MCMC_TUNE,
        draw_steps=MCMC_DRAWS,
    )

    if results_df.empty:
        print("❌ CRITICAL ERROR: Parameter estimation results summary is empty.")
        return

    # 3. Extract and log real MCMC convergence diagnostics when available
    print("\n[Diagnostics] Checking MCMC convergence metrics (R-hat & ESS)...")
    convergence_info = _extract_trace_diagnostics(results_df)
    if convergence_info["max_r_hat"] is not None:
        print(
            f" > max R-hat: {convergence_info['max_r_hat']:.4f} | "
            f"min ESS bulk: {convergence_info['min_ess_bulk']:.1f} | "
            f"divergences: {convergence_info['divergent_transitions']}"
        )
    else:
        print(
            " > No ArViZ InferenceData trace exposed by visualization/results_df; "
            "marking convergence status as unverified in metadata."
        )

    # 4. Execute Predictive Forecasting Layer
    print("\n[Step 3/4] Running future tumor growth forecasting engine...")
    cohort_forecasts = run_cohort_forecast(
        results_summary_df=results_df,
        df_vol=df_vol,
        forecast_days=FORECAST_DAYS,
    )

    # 5. Save final estimated parameters, growth forecasts, volume history & provenance metadata
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    out_csv = output_dir / "hbm_patient_parameters_summary.csv"
    results_df.to_csv(out_csv, index=False)

    forecast_combined = (
        pd.concat(cohort_forecasts.values(), ignore_index=True)
        if cohort_forecasts
        else pd.DataFrame()
    )
    forecast_csv = output_dir / "vht_tumor_growth_forecasts.csv"
    forecast_combined.to_csv(forecast_csv, index=False)

    vol_csv = output_dir / "df_vol.csv"
    df_vol.to_csv(vol_csv, index=False)

    metadata = {
        "execution_timestamp": datetime.datetime.now().isoformat(),
        "cohort_size_patients": int(results_df["Patient_ID"].nunique()),
        "observables": {
            "volume_total_present": "Volume_Total" in df_vol.columns,
            "volume_invasive_present": has_invasive_col,
        },
        "library_versions": {
            "pymc": pm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nibabel": nib.__version__,
            "arviz": az.__version__,
        },
        "mcmc_hyperparameters": {
            "draws": MCMC_DRAWS,
            "tune": MCMC_TUNE,
            "target_accept_configured_in_main": TARGET_ACCEPT,
            "random_seed_configured_in_main": RANDOM_SEED,
            "forecast_days": FORECAST_DAYS,
        },
        "convergence_verification": convergence_info,
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