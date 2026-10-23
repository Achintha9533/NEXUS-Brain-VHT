"""
NEXUS Brain VHT: Predictive Forecasting & Counterfactual Simulation Module
=========================================================================
Projects future glioblastoma tumor growth trajectories beyond historical clinical 
timepoints using patient-specific calibrated Fisher-KPP parameters, supporting 
both standard baseline forecasts and dynamic counterfactual "What-If" simulations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_patient_trajectory(
    patient_id: str,
    last_day: float,
    last_volume: float,
    rho: float,
    D: float,
    K: float,
    treatment_params: dict,
    forecast_days: int = 180,
    dt: float = 1.0
) -> pd.DataFrame:
    """
    Simulates future tumor volume trajectories iteratively using the Fisher-KPP 
    reaction-diffusion wavefront model with static treatment flags.
    """
    days = np.arange(0, forecast_days + dt, dt)
    volumes = []
    
    # Initialize radius from last known total volume (spherical approximation)
    r_curr = (3.0 * last_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    for step_day in days:
        # Calculate current equivalent volume
        v_curr = (4.0 / 3.0) * np.pi * (r_curr ** 3.0)
        volumes.append(v_curr)
        
        # Carrying capacity saturation factor
        logistic_capacity = max(1.0 - (v_curr / K), 1e-4)
        
        agents = ["Chemo", "Rad", "Immuno", "Additional1", "Additional2", "Brachy"]
        active_damping = sum(
            treatment_params.get(agent, 0.0) * treatment_params.get(f"{agent}_active_flag", 0.0)
            for agent in agents
        ) if isinstance(treatment_params, dict) else 0.0
        
        treatment_damping = np.exp(-active_damping)
        
        # Effective reaction rate
        rho_eff = max(rho * logistic_capacity * treatment_damping, 1e-8)
        
        # Fisher-KPP Wavefront Velocity: v = 2 * sqrt(D * rho_eff)
        v_fisher = 2.0 * np.sqrt(max(D * rho_eff, 0.0))
        
        # Advance radius forward by step dt
        r_curr += dt * v_fisher

    forecast_df = pd.DataFrame({
        "Patient_ID": patient_id,
        "Forecast_Day": last_day + days,
        "Forecast_Volume": volumes
    })
    
    return forecast_df


def simulate_counterfactual_scenario(
    patient_id: str,
    last_day: float,
    last_volume: float,
    rho: float,
    D: float,
    K: float,
    calibrated_alphas: dict,
    treatment_schedule: list[dict],
    resection_events: list[dict] = None,
    forecast_days: int = 180,
    dt: float = 1.0
) -> pd.DataFrame:
    """
    Simulates counterfactual scenarios where treatment windows or surgical resection events
    occur dynamically at specific future absolute days.
    """
    days = np.arange(0, forecast_days + dt, dt)
    volumes = []
    
    r_curr = (3.0 * last_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    for step_day in days:
        current_absolute_day = last_day + step_day
        
        # Check for surgical resection events on this day
        if resection_events:
            for event in resection_events:
                if abs(current_absolute_day - event["day"]) < (dt / 2.0):
                    current_vol = (4.0 / 3.0) * np.pi * (r_curr ** 3.0)
                    new_vol = current_vol * (1.0 - event["volume_reduction_fraction"])
                    r_curr = (3.0 * new_vol / (4.0 * np.pi)) ** (1.0 / 3.0)

        v_curr = (4.0 / 3.0) * np.pi * (r_curr ** 3.0)
        volumes.append(v_curr)
        
        logistic_capacity = max(1.0 - (v_curr / K), 1e-4)
        
        # Calculate active damping dynamically based on current absolute day schedule
        active_damping = 0.0
        if treatment_schedule:
            for sched in treatment_schedule:
                if sched["start_day"] <= current_absolute_day <= sched["end_day"]:
                    agent = sched["agent"]
                    alpha = calibrated_alphas.get(agent, 0.0)
                    active_damping += alpha * sched.get("active_flag", 1.0)
                
        treatment_damping = np.exp(-active_damping)
        rho_eff = max(rho * logistic_capacity * treatment_damping, 1e-8)
        v_fisher = 2.0 * np.sqrt(max(D * rho_eff, 0.0))
        
        r_curr += dt * v_fisher

    forecast_df = pd.DataFrame({
        "Patient_ID": patient_id,
        "Forecast_Day": last_day + days,
        "Forecast_Volume": volumes
    })
    
    return forecast_df


def run_cohort_forecast(
    results_summary_df: pd.DataFrame, 
    df_vol: pd.DataFrame, 
    forecast_days: int = 180
) -> dict:
    """
    Runs forward projections across all calibrated patients in the cohort.
    """
    print(f"\nLaunching VHT Predictive Forecasting Engine ({forecast_days}-day horizon)...")
    forecasts = {}

    for _, row in results_summary_df.iterrows():
        pid = row["Patient_ID"]
        rho = row["rho"]
        D = row["D"]
        K = row.get("K", 150000.0)
        
        p_history = df_vol[df_vol["Patient_ID"] == pid].sort_values("Day")
        if p_history.empty:
            continue
            
        last_row = p_history.iloc[-1]
        last_day = float(last_row["Day"])
        last_volume = float(last_row["Volume_Total"])
        
        t_params = {
            "Chemo": row.get("Chemo", 0.0),
            "Rad": row.get("Rad", 0.0),
            "Immuno": row.get("Immuno", 0.0),
            "Additional1": row.get("Additional1", 0.0),
            "Additional2": row.get("Additional2", 0.0),
            "Brachy": row.get("Brachy", 0.0),
            "Chemo_active_flag": 0.0,
            "Rad_active_flag": 0.0,
        }

        f_df = forecast_patient_trajectory(
            patient_id=pid,
            last_day=last_day,
            last_volume=last_volume,
            rho=rho,
            D=D,
            K=K,
            treatment_params=t_params,
            forecast_days=forecast_days
        )
        forecasts[pid] = f_df

    print(f"Successfully generated forecasts for {len(forecasts)} patients.")
    return forecasts