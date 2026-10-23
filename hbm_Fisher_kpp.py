"""
Hierarchical Bayesian Modeling (HBM) for Glioblastoma Growth Parameter Estimation
=============================================================================
Implements a Hierarchical Bayesian model using PyMC and PyTensor based on the 
Fisher-KPP Reaction-Diffusion equation:
    dc/dt = D * grad^2(c) + rho * c * (1 - c/K) - Kill(t)*c
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt


def build_hierarchical_fisher_kpp_model(cohort_df: pd.DataFrame) -> pm.Model:
    """
    Constructs a PyMC Hierarchical Bayesian model graph for longitudinal tumor 
    growth prediction and treatment effect estimation using the Fisher-KPP equation.
    """
    df_local = cohort_df.copy()

    # Re-index Patient_Idx to ensure contiguous indices (0 ... N-1)
    unique_pids = df_local["Patient_ID"].unique()
    id_map = {pid: idx for idx, pid in enumerate(unique_pids)}
    df_local["Patient_Idx"] = df_local["Patient_ID"].map(id_map)

    # Extract arrays
    p_idx_arr = df_local["Patient_Idx"].values.astype(int)
    num_patients = int(p_idx_arr.max() + 1)

    dt_arr = df_local["dt"].values.astype(np.float64)
    is_base_arr = df_local["Is_Baseline"].values.astype(np.float64)
    obs_vol_arr = df_local["Volume_Total"].values.astype(np.float64)

    # Active treatment exposure arrays (0.0 to 1.0 fraction of time active)
    chemo_act  = df_local.get("Chemo_ActiveFraction", df_local.get("Chemo_Active", 0.0)).values.astype(np.float64)
    rad_act    = df_local.get("Rad_ActiveFraction", df_local.get("Rad_Active", 0.0)).values.astype(np.float64)
    immuno_act = df_local.get("Immuno_ActiveFraction", df_local.get("Immuno_Active", 0.0)).values.astype(np.float64)
    add1_act   = df_local.get("Additional1_ActiveFraction", df_local.get("Additional1_Active", 0.0)).values.astype(np.float64)
    add2_act   = df_local.get("Additional2_ActiveFraction", df_local.get("Additional2_Active", 0.0)).values.astype(np.float64)
    brachy_act = df_local.get("Brachy_ActiveFraction", df_local.get("Brachy_Active", 0.0)).values.astype(np.float64)

    with pm.Model() as model:
        # =====================================================================
        # 1. POPULATION HYPER-PRIORS (Literature-Grounded)
        # =====================================================================
        mu_log_rho = pm.Normal("mu_log_rho", mu=-4.5, sigma=1.5)
        mu_log_D   = pm.Normal("mu_log_D",   mu=-3.0, sigma=1.5)
        sigma_log_rho = pm.HalfNormal("sigma_log_rho", sigma=1.0)
        sigma_log_D   = pm.HalfNormal("sigma_log_D",   sigma=1.0)

        # Carrying capacity hyper-priors
        mu_log_K    = pm.Normal("mu_log_K", mu=11.5, sigma=1.0)
        sigma_log_K = pm.HalfNormal("sigma_log_K", sigma=0.5)

        # Treatment inhibition hyper-priors
        mu_alpha_chemo  = pm.HalfNormal("mu_alpha_chemo",  sigma=1.0)
        mu_alpha_rad    = pm.HalfNormal("mu_alpha_rad",    sigma=1.0)
        mu_alpha_immuno = pm.HalfNormal("mu_alpha_immuno", sigma=1.0)
        mu_alpha_add1   = pm.HalfNormal("mu_alpha_add1",   sigma=1.0)
        mu_alpha_add2   = pm.HalfNormal("mu_alpha_add2",   sigma=1.0)
        mu_alpha_brachy = pm.HalfNormal("mu_alpha_brachy", sigma=1.0)

        sigma_obs_log_v = pm.HalfNormal("sigma_obs_log_v", sigma=0.25)

        # =====================================================================
        # 2. PATIENT-LEVEL PARAMETERS (Non-Centered Parameterization)
        # =====================================================================
        rho_off = pm.Normal("rho_offset", shape=num_patients)
        D_off   = pm.Normal("D_offset",   shape=num_patients)
        K_off   = pm.Normal("K_offset",   shape=num_patients)

        rho = pm.Deterministic("rho", pt.exp(mu_log_rho + rho_off * sigma_log_rho)) # 1/day
        D   = pm.Deterministic("D",   pt.exp(mu_log_D   + D_off   * sigma_log_D))   # mm^2/day
        K   = pm.Deterministic("K",   pt.exp(mu_log_K   + K_off   * sigma_log_K))   # mm^3

        alpha_chemo  = pm.HalfNormal("alpha_chemo",  sigma=mu_alpha_chemo,  shape=num_patients)
        alpha_rad    = pm.HalfNormal("alpha_rad",    sigma=mu_alpha_rad,    shape=num_patients)
        alpha_immuno = pm.HalfNormal("alpha_immuno", sigma=mu_alpha_immuno, shape=num_patients)
        alpha_add1   = pm.HalfNormal("alpha_add1",   sigma=mu_alpha_add1,   shape=num_patients)
        alpha_add2   = pm.HalfNormal("alpha_add2",   sigma=mu_alpha_add2,   shape=num_patients)
        alpha_brachy = pm.HalfNormal("alpha_brachy", sigma=mu_alpha_brachy, shape=num_patients)

        # Map patient parameters to longitudinal timepoint rows
        row_rho      = rho[p_idx_arr]
        row_D        = D[p_idx_arr]
        row_K        = K[p_idx_arr]
        row_a_chemo  = alpha_chemo[p_idx_arr]
        row_a_rad    = alpha_rad[p_idx_arr]
        row_a_imm    = alpha_immuno[p_idx_arr]
        row_a_add1   = alpha_add1[p_idx_arr]
        row_a_add2   = alpha_add2[p_idx_arr]
        row_a_brachy = alpha_brachy[p_idx_arr]

        # =====================================================================
        # 3. FISHER-KPP STEP KINETICS
        # =====================================================================
        def fisher_kpp_step(
            dt, is_baseline, obs_vol, rho_p, d_p, k_p,
            ch_a, rd_a, im_a, ad1_a, ad2_a, br_a,
            a_ch, a_rd, a_im, a_ad1, a_ad2, a_br,
            prev_r
        ):
            # Baseline radius initialization (assuming spherical expansion: r = (3V / 4pi)^(1/3))
            r_base = (3.0 * obs_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
            r_curr = pt.switch(pt.eq(is_baseline, 1.0), r_base, prev_r)

            # Equivalent current volume
            v_curr = (4.0 / 3.0) * np.pi * (r_curr ** 3.0)

            # Carrying capacity saturation factor
            logistic_capacity = pt.maximum(1.0 - (v_curr / k_p), 1e-4)

            # Damping factor acting on effective proliferation
            treatment_damping = pt.exp(
                -(a_ch * ch_a + a_rd * rd_a + a_im * im_a + 
                  a_ad1 * ad1_a + a_ad2 * ad2_a + a_br * br_a)
            )

            # Effective reaction rate
            rho_eff = pt.maximum(rho_p * logistic_capacity * treatment_damping, 1e-8)

            # Fisher-KPP Wavefront Velocity: v = 2 * sqrt(D * rho_eff)
            v_fisher = 2.0 * pt.sqrt(d_p * rho_eff)

            # Radial advance over step interval dt
            r_next = r_curr + (dt * v_fisher)

            return r_next

        # Scan across longitudinal timepoints
        predicted_radii, _ = pytensor.scan(
            fn=fisher_kpp_step,
            sequences=[
                pt.as_tensor_variable(dt_arr, dtype="float64"),
                pt.as_tensor_variable(is_base_arr, dtype="float64"),
                pt.as_tensor_variable(obs_vol_arr, dtype="float64"),
                row_rho,
                row_D,
                row_K,
                pt.as_tensor_variable(chemo_act, dtype="float64"),
                pt.as_tensor_variable(rad_act, dtype="float64"),
                pt.as_tensor_variable(immuno_act, dtype="float64"),
                pt.as_tensor_variable(add1_act, dtype="float64"),
                pt.as_tensor_variable(add2_act, dtype="float64"),
                pt.as_tensor_variable(brachy_act, dtype="float64"),
                row_a_chemo,
                row_a_rad,
                row_a_imm,
                row_a_add1,
                row_a_add2,
                row_a_brachy,
            ],
            outputs_info=[pt.as_tensor_variable(0.0, dtype="float64")]
        )

        pred_volumes = pm.Deterministic(
            "pred_volumes", 
            (4.0 / 3.0) * np.pi * (predicted_radii ** 3.0)
        )

        # Likelihood over log-transformed volume measurements
        pm.Normal(
            "likelihood",
            mu=pt.log(pt.maximum(pred_volumes, 1.0)),
            sigma=sigma_obs_log_v,
            observed=np.log(np.maximum(obs_vol_arr, 1.0))
        )

    return model