"""
Interactive Cohort Visualization Layer for Hierarchical Bayesian Modeling (HBM)
=============================================================================
Compiles records globally, runs MCMC sampling via PyMC, extracts posterior metrics,
and plots longitudinal patient trajectories with treatment window shading.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # <--- Disables GUI pop-up blocks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# Import model builder and data prep modules from your repository
from vht_pipeline import build_patient_volume_dataframe
from hbm_Fisher_kpp import build_hierarchical_fisher_kpp_model


def estimate_and_plot_each_patient(
    patient_objects: dict, 
    tune_steps: int = 1000, 
    draw_steps: int = 1000
) -> pd.DataFrame:
    """
    Unified HBM pipeline that compiles records globally and streams posterior metrics 
    directly down into comparative timeline plots.
    """
    df = build_patient_volume_dataframe(patient_objects)
    
    if df.empty:
        print("❌ CRITICAL ERROR: build_patient_volume_dataframe returned an EMPTY DataFrame!")
        return pd.DataFrame()

    print(f"Initializing Structural HBM network across {df['Patient_Idx'].nunique()} patient records...")
    hbm_model = build_hierarchical_fisher_kpp_model(df)
    
    print("Launching global NUTS Hamiltonian sampling chains...")
    with hbm_model:
        trace = pm.sample(draws=draw_steps, tune=tune_steps, target_accept=0.90, random_seed=42)
    
    # Extract mean structural configurations across the tracking posterior context
    post_summary = az.summary(trace, kind="stats")
    
    # Fetch predicted trajectories calculated directly by PyMC's tensor graph layers
    mean_predictions = trace.posterior["pred_volumes"].mean(dim=["chain", "draw"]).values

    treatment_names = ["Chemo", "Rad", "Additional1", "Additional2", "Immuno", "Brachy"]
    burn_colors = {
        "Chemo": "tab:green", "Rad": "tab:orange", 
        "Additional1": "tab:purple", "Additional2": "tab:pink", 
        "Immuno": "tab:cyan", "Brachy": "tab:olive"
    }

    keys_to_check = [
        ("Chemo", "Number of days from Diagnosis to Initial Chemo Therapy Start date", "Number of days from Diagnosis to Initial Chemo Therapy end date"),
        ("Rad", "Number of days from Diagnosis to Radiation Therapy Start date", "Number of days from Diagnosis to Radiation Therapy end date"),
        ("Additional1", "Number of Days from Diagnosis to Starting Additional Therapy", "Number of Days from Diagnosis to Complete Additional Therapy"),
        ("Additional2", "Number of Days from Diagnosis to Starting 2nd_Additional Therapy", "Number of Days from Diagnosis to Complete 2nd_Additional Therapy"),
        ("Immuno", "Number of Days from Diagnosis to Start Immunotherapy", "Number of Days from Diagnosis to Complete Immunotherapy"),
        ("Brachy", "Number of Days from Diagnosis to the day of Insertion of Brachytherapy", "Number of Days from Diagnosis to the day of Insertion of Brachytherapy")
    ]

    results = []

    for p_idx, (pid, g) in enumerate(df.groupby("Patient_ID")):
        g = g.sort_values("Day").reset_index(drop=True)
        row_indices = g.index.values + (df[df["Patient_ID"] == pid].index[0])
        
        # Pull specific parameter point statistics out of the unified tracking summary
        rho_mean = float(post_summary.loc[f"rho[{p_idx}]", "mean"])
        D_mean = float(post_summary.loc[f"D[{p_idx}]", "mean"])
        
        # Safely compute wave speed (v = 2*sqrt(D*rho)) directly or fetch from trace
        if f"v_front[{p_idx}]" in post_summary.index:
            v_mean = float(post_summary.loc[f"v_front[{p_idx}]", "mean"])
        else:
            v_mean = float(2.0 * np.sqrt(D_mean * rho_mean))
        
        # Calculate localized deceleration efficacy percentages for all 6 therapy arms
        tparams = {name: np.nan for name in treatment_names}
        tparams["Chemo"]       = float(np.exp(-post_summary.loc[f"alpha_chemo[{p_idx}]", "mean"]))
        tparams["Rad"]         = float(np.exp(-post_summary.loc[f"alpha_rad[{p_idx}]", "mean"]))
        tparams["Immuno"]      = float(np.exp(-post_summary.loc[f"alpha_immuno[{p_idx}]", "mean"]))
        tparams["Additional1"] = float(np.exp(-post_summary.loc[f"alpha_add1[{p_idx}]", "mean"]))
        tparams["Additional2"] = float(np.exp(-post_summary.loc[f"alpha_add2[{p_idx}]", "mean"]))
        tparams["Brachy"]      = float(np.exp(-post_summary.loc[f"alpha_brachy[{p_idx}]", "mean"]))

        results.append({
            "Patient_ID": pid, "rho": rho_mean, "D": D_mean, "wave_speed": v_mean, **tparams
        })

        track_strings = []
        for k in treatment_names:
            val = tparams[k]
            if pd.notna(val):
                reduction = (1.0 - val) * 100
                track_strings.append(f"{k}_Effectiveness = {reduction:.1f}%")
            else:
                track_strings.append(f"{k}_Effectiveness = not modeled")
                
        print(f"Patient {pid}: rho = {rho_mean:.5f}, D = {D_mean:.5f} | " + ", ".join(track_strings))

        # --- PLOTTING GENERATION ---
        fig, ax = plt.subplots(figsize=(9, 5))
        obj = patient_objects[pid]
        clinical = obj.get("clinical", {}) if isinstance(obj, dict) else {}
        max_day_present = g["Day"].max()
        
        # 1. Plot Background Shading Boxes
        for name, start_key, end_key in keys_to_check:
            s_val = pd.to_numeric(clinical.get(start_key, np.nan), errors="coerce")
            e_val = pd.to_numeric(clinical.get(end_key, np.nan), errors="coerce")
            
            if pd.notna(s_val):
                start_day = max(0.0, float(s_val))
                end_day = float(e_val) if pd.notna(e_val) else max_day_present
                if end_day <= start_day:
                    end_day = start_day + 2.0 
                ax.axvspan(start_day, end_day, color=burn_colors[name], alpha=0.15, label=f"Active Window: {name}")

        # 2. Plot Observation points
        ax.plot(g["Day"], g["Volume_Total"], marker="o", linewidth=2, color="black", label="Total volume")
        if g["Volume_Core"].notna().any():
            ax.plot(g["Day"], g["Volume_Core"], marker="s", linewidth=2, color="tab:blue", label="Core volume")

        # 3. Plot Differentiable Tensor Model Fit Trajectory
        ax.plot(g["Day"], mean_predictions[row_indices], "-", color="tab:red", linewidth=2.5, label="HBM Posterior Mean Trajectory")

        ax.set_title(f"Patient {pid} | HBM Profile | rho={rho_mean:.4g} | D={D_mean:.4g}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Volume ($mm^3$)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()

    return pd.DataFrame(results)