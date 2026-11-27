"""
Interactive Cohort Visualization Layer for Hierarchical Bayesian Modeling (HBM)
=============================================================================
Compiles records globally, runs MCMC sampling via PyMC, extracts posterior metrics,
plots longitudinal patient trajectories with treatment window shading, and
automatically exports MCMC convergence diagnostic figures for thesis inclusion.
"""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")  # Disables GUI pop-up blocks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# Import model builder, data prep modules, and helper configurations from repository
from vht_pipeline import build_patient_volume_dataframe
from hbm_Fisher_kpp import build_hierarchical_fisher_kpp_model
from vht_helper_functions import keys_to_check

latest_trace = None


def estimate_and_plot_each_patient(
    patient_objects: dict,
    tune_steps: int = 1000,
    draw_steps: int = 1000,
    target_accept: float = 0.90,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Unified HBM pipeline that compiles records globally, runs MCMC sampling,
    exports diagnostic plots, and extracts posterior parameter summaries.
    """
    global latest_trace

    df = build_patient_volume_dataframe(patient_objects)

    if df.empty:
        print("❌ CRITICAL ERROR: build_patient_volume_dataframe returned an EMPTY DataFrame!")
        return pd.DataFrame()

    df = df.copy()
    unique_pids = df["Patient_ID"].unique()
    id_map = {pid: idx for idx, pid in enumerate(unique_pids)}
    df["Patient_Idx"] = df["Patient_ID"].map(id_map).astype(int)

    n_patients = len(unique_pids)
    print(f"Initializing Structural HBM network across {n_patients} patient records...")
    hbm_model = build_hierarchical_fisher_kpp_model(df)

    print("Launching global NUTS Hamiltonian sampling chains...")
    with hbm_model:
        trace = pm.sample(
            draws=draw_steps,
            tune=tune_steps,
            target_accept=target_accept,
            random_seed=random_seed,
        )

    latest_trace = trace

    hyper_vars = [
        "mu_log_prod",
        "mu_log_ratio",
        "mu_log_rho",
        "mu_log_D",
        "mu_log_K",
        "sigma_obs_log_v",
        "sigma_obs_log_inv",
    ]
    present_hyper_vars = [v for v in hyper_vars if v in trace.posterior.data_vars]

    # Compute full summary once to avoid redundant sampling-graph reductions
    full_summary = az.summary(trace, kind="all")

    # --- AUTOMATED MCMC CONVERGENCE DIAGNOSTIC EXPORT & REPORTING ---
    print("\n" + "=" * 75)
    print("      MCMC CONVERGENCE DIAGNOSTICS (ARVIZ R-hat & ESS)      ")
    print("=" * 75)
    try:
        diag_summary = full_summary.loc[
            [v for v in present_hyper_vars if v in full_summary.index]
        ]
        print(diag_summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]])

        divergences = int(np.asarray(trace.sample_stats.diverging.values).sum())
        print(f"\nTotal Divergent Transitions: {divergences}")

        os.makedirs("figures", exist_ok=True)
        plot_vars = [
            v
            for v in ["mu_log_prod", "mu_log_ratio", "mu_log_rho", "mu_log_D", "mu_log_K"]
            if v in trace.posterior.data_vars
        ]

        print("\n > Exporting MCMC convergence plots to 'figures/' directory...")
        az.plot_trace(trace, var_names=plot_vars)
        plt.suptitle("MCMC Trace Plots for Fisher-KPP Hyperpriors", y=1.02, fontsize=14)
        plt.savefig("figures/mcmc_trace_plots.png", dpi=300, bbox_inches="tight")
        plt.close()

        az.plot_rank(trace, var_names=plot_vars)
        plt.suptitle("MCMC Rank Plots for Fisher-KPP Hyperpriors", y=1.02, fontsize=14)
        plt.savefig("figures/mcmc_rank_plots.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(" > Successfully saved: figures/mcmc_trace_plots.png, figures/mcmc_rank_plots.png")

    except Exception as e:
        print(f"Diagnostic reporting / plotting note: {e}")
    print("=" * 75 + "\n")

    # --- ENHANCED STATISTICAL SUMMARY REPORTING ---
    print("\n" + "=" * 75)
    print("      HIERARCHICAL BAYESIAN MODEL (HBM) POPULATION HYPER-PRIORS      ")
    print("      (Posterior Mean & 94% HDI Confidence Intervals)                ")
    print("=" * 75)

    pop_summary = full_summary.loc[
        [v for v in present_hyper_vars if v in full_summary.index]
    ]

    print(f"{'Parameter':<20} | {'Mean':<10} | {'SD':<10} | {'HDI 3%':<10} | {'HDI 97%':<10}")
    print("-" * 68)

    for param_name, row in pop_summary.iterrows():
        mean_val = float(row.get("mean", 0.0))
        sd_val = float(row.get("sd", 0.0))
        hdi_3 = float(row.get("hdi_3%", row.get("hdi_5.5%", 0.0)))
        hdi_97 = float(row.get("hdi_97%", row.get("hdi_94.5%", 0.0)))
        print(f"{param_name:<20} | {mean_val:<10.4f} | {sd_val:<10.4f} | {hdi_3:<10.4f} | {hdi_97:<10.4f}")
    print("=" * 75 + "\n")

    mean_predictions = trace.posterior["pred_volumes"].mean(dim=["chain", "draw"]).values

    treatment_names = ["Chemo", "Rad", "Additional1", "Additional2", "Immuno", "Brachy"]
    alpha_var_map = {
        "Chemo": "alpha_chemo",
        "Rad": "alpha_rad",
        "Immuno": "alpha_immuno",
        "Additional1": "alpha_add1",
        "Additional2": "alpha_add2",
        "Brachy": "alpha_brachy",
    }
    burn_colors = {
        "Chemo": "tab:green",
        "Rad": "tab:orange",
        "Additional1": "tab:purple",
        "Additional2": "tab:pink",
        "Immuno": "tab:cyan",
        "Brachy": "tab:olive",
    }

    results = []

    for pid in unique_pids:
        p_idx = id_map[pid]
        g = df[df["Patient_ID"] == pid].sort_values("Day")

        rho_mean = float(full_summary.loc[f"rho[{p_idx}]", "mean"])
        D_mean = float(full_summary.loc[f"D[{p_idx}]", "mean"])
        v_mean = float(2.0 * np.sqrt(max(D_mean * rho_mean, 0.0)))

        tparams = {name: np.nan for name in treatment_names}
        for name, base_var in alpha_var_map.items():
            key = f"{base_var}[{p_idx}]"
            if key in full_summary.index:
                tparams[name] = float(np.exp(-full_summary.loc[key, "mean"]))

        results.append(
            {
                "Patient_ID": pid,
                "rho": rho_mean,
                "D": D_mean,
                "wave_speed": v_mean,
                **tparams,
            }
        )

        track_strings = []
        for k in treatment_names:
            val = tparams[k]
            if pd.notna(val):
                reduction = (1.0 - val) * 100.0
                track_strings.append(f"{k}_Effectiveness = {reduction:.1f}%")
            else:
                track_strings.append(f"{k}_Effectiveness = not modeled")

        print(f"Patient {pid}: rho = {rho_mean:.5f}, D = {D_mean:.5f} | " + ", ".join(track_strings))

        # --- PLOTTING GENERATION ---
        fig, ax = plt.subplots(figsize=(9, 5))
        obj = patient_objects.get(pid, {})
        clinical = obj.get("clinical", {}) if isinstance(obj, dict) else {}
        max_day_present = g["Day"].max()

        for name, start_key, end_key in keys_to_check:
            s_val = pd.to_numeric(clinical.get(start_key, np.nan), errors="coerce")
            e_val = pd.to_numeric(clinical.get(end_key, np.nan), errors="coerce")

            if pd.notna(s_val):
                start_day = max(0.0, float(s_val))
                end_day = float(e_val) if pd.notna(e_val) else float(max_day_present)
                if end_day <= start_day:
                    end_day = start_day + 2.0
                ax.axvspan(
                    start_day,
                    end_day,
                    color=burn_colors.get(name, "gray"),
                    alpha=0.15,
                    label=f"Active Window: {name}",
                )

        ax.plot(g["Day"], g["Volume_Total"], marker="o", linewidth=2, color="black", label="Total volume")
        if "Volume_Core" in g.columns and g["Volume_Core"].notna().any():
            ax.plot(g["Day"], g["Volume_Core"], marker="s", linewidth=2, color="tab:blue", label="Core volume")

        # Map sorted patient rows back to positional indices in df (which match pred_volumes)
        pos_indices = df.index.get_indexer(g.index)
        if len(pos_indices) == len(g) and np.all(pos_indices >= 0):
            ax.plot(
                g["Day"],
                mean_predictions[pos_indices],
                "-",
                color="tab:red",
                linewidth=2.5,
                label="HBM Posterior Mean Trajectory",
            )

        ax.set_title(f"Patient {pid} | HBM Profile | rho={rho_mean:.4g} | D={D_mean:.4g}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Volume ($mm^3$)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.close(fig)

    results_df = pd.DataFrame(results)
    results_df.attrs["trace"] = trace
    return results_df