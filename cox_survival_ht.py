import vht_pipeline
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Import all helper functions and configurations from vht_helper_functions.py
from vht_helper_functions import *

from lifelines import CoxPHFitter
HAS_LIFELINES = True

def build_survival_dataset(patient_objects, df_volumes, df_kinetics):
    survival_rows = []

    for patient_id, obj in patient_objects.items():
        clean_id = obj.get("PID_Clean", str(patient_id).split("-")[0])
        clinical = obj.get("clinical", {})

        # Corrected column lookup for survival duration
        duration = (
            get_numeric(clinical, "Number of days from Diagnosis to death (Days)") or
            get_numeric(clinical, "Overall Survival (Days)") or
            get_numeric(clinical, "OS_Days") or
            get_numeric(clinical, "Survival Time") or
            get_numeric(clinical, "Days to Death or Last Followup")
        )

        # Corrected column lookup for event status (0 or 1)
        event_val = clinical.get("Overall Survival (Death)")
        if pd.notna(event_val):
            event = int(float(event_val))
        else:
            status_raw = str(clinical.get("Vital Status", clinical.get("Status", ""))).upper()
            event = 1 if any(word in status_raw for word in ["DEAD", "DECEASED", "EXPIRED", "1", "TRUE"]) else 0

        # If duration is unlisted/NaN (e.g. censored patients), fallback to last known imaging day
        if pd.isna(duration) or duration <= 0:
            p_vols = df_volumes[df_volumes["Patient_ID"] == patient_id]
            if not p_vols.empty:
                duration = float(p_vols["Day"].max())
            else:
                continue

        p_vols = df_volumes[df_volumes["Patient_ID"] == patient_id].sort_values("Day")
        if p_vols.empty:
            continue
            
        base_vol = p_vols.iloc[0]
        max_vol = p_vols["Volume_Total"].max()

        p_kin = df_kinetics[df_kinetics["PID_Clean"] == clean_id] if df_kinetics is not None and not df_kinetics.empty else pd.DataFrame()
        mean_enhancing_vel = p_kin["Vel_Enhancing"].mean() if not p_kin.empty and "Vel_Enhancing" in p_kin.columns else 0.0

        patient_age = (
            get_numeric(clinical, "Age at diagnosis") or
            get_numeric(clinical, "Age") or
            get_numeric(clinical, "Patient Age")
        )

        s_row = {
            "Patient_ID": patient_id,
            "Duration": float(duration),
            "Event": int(event),
            "Baseline_Volume_Total": float(base_vol["Volume_Total"]),
            "Baseline_Volume_Enhancing": float(base_vol["Volume_Enhancing"]),
            "Peak_Volume_Total": float(max_vol),
            "Mean_Enhancing_Velocity": float(mean_enhancing_vel) if not np.isnan(mean_enhancing_vel) else 0.0,
            "Age": float(patient_age) if pd.notna(patient_age) else np.nan,
        }

        for marker in biological_priors:
            s_row[marker] = clinical.get(marker, 0)

        survival_rows.append(s_row)

    df_survival = pd.DataFrame(survival_rows)

    if not df_survival.empty:
        if "Age" in df_survival.columns:
            median_age = df_survival["Age"].median()
            df_survival["Age"] = df_survival["Age"].fillna(median_age if not np.isnan(median_age) else 60.0)

        for col in biological_priors:
            if col in df_survival.columns:
                df_survival[col] = df_survival[col].fillna(0)
                if df_survival[col].dtype == "object":
                    df_survival[col] = LabelEncoder().fit_transform(df_survival[col].astype(str))

    return df_survival


def run_cox_survival_analysis(df_survival):
    print("\n=========================================================")
    print("      COX PROPORTIONAL HAZARDS SURVIVAL ANALYSIS         ")
    print("=========================================================")

    if not HAS_LIFELINES:
        print("❌ 'lifelines' library is not installed. Run 'pip install lifelines' to run full fit.")
        return None

    feature_cols = [
        "Duration", "Event", 
        "Baseline_Volume_Total", "Baseline_Volume_Enhancing", 
        "Mean_Enhancing_Velocity", "Age"
    ]
    
    for marker in ["IDH1 mutation", "MGMT methylation", "1p/19q"]:
        if marker in df_survival.columns:
            feature_cols.append(marker)

    cox_data = df_survival[feature_cols].copy()

    # Impute missing values with medians
    for col in cox_data.columns:
        if cox_data[col].isna().any():
            median_val = cox_data[col].median()
            cox_data[col] = cox_data[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    cox_data = cox_data.dropna(subset=["Duration", "Event"])

    # Remove columns with zero variance (constant values) to avoid LinAlgWarning & NaN delta errors
    valid_cols = ["Duration", "Event"]
    for col in cox_data.columns:
        if col not in ["Duration", "Event"]:
            if cox_data[col].nunique() > 1:
                valid_cols.append(col)
            else:
                print(f"⚠️ Skipping column '{col}' due to zero variance (constant values).")

    cox_data = cox_data[valid_cols]

    # Normalize/Scale continuous features to prevent matrix ill-conditioning
    scale_cols = [c for c in cox_data.columns if c not in ["Duration", "Event"]]
    for col in scale_cols:
        col_std = cox_data[col].std()
        if col_std > 0:
            cox_data[col] = (cox_data[col] - cox_data[col].mean()) / col_std

    if len(cox_data) == 0:
        print("❌ Error: Cox dataset is empty after processing.")
        return None

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_data, duration_col="Duration", event_col="Event")
        
        # --- ENHANCED TABULAR STATISTICAL SUMMARY REPORTING ---
        print("\n" + "="*85)
        print("          COX PROPORTIONAL HAZARDS SURVIVAL ANALYSIS SUMMARY       ")
        print("="*85)
        
        summary_df = cph.summary
        formatted_summary = pd.DataFrame({
            "Coef (beta)": summary_df["coef"].round(4),
            "Exp(coef) [HR]": summary_df["exp(coef)"].round(4),
            "Std Error": summary_df["se(coef)"].round(4),
            "p-value": summary_df["p"].apply(lambda p: f"{p:.4e}" if p < 0.001 else f"{p:.4f}"),
            "Lower 95%": summary_df["exp(coef) lower 95%"].round(4),
            "Upper 95%": summary_df["exp(coef) upper 95%"].round(4),
        })
        
        print(formatted_summary.to_string())
        print("="*85)
        print(f" Concordance Index (C-index): {cph.concordance_index_:.4f}")
        print(f" Log-Likelihood Ratio Test p-value: {cph.log_likelihood_ratio_test().p_value:.4e}")
        print("="*85 + "\n")
        
        return cph
    except Exception as e:
        print(f"❌ Error fitting Cox Model: {e}")
        return None


def main():
    patient_objects, df_kinetics, df_volumes, clinical_df = vht_pipeline.run_pipeline()
    df_survival = build_survival_dataset(patient_objects, df_volumes, df_kinetics)
    print(f"\nBuilt survival dataset for {len(df_survival)} patients.")

    cph_model = run_cox_survival_analysis(df_survival)
    return patient_objects, df_kinetics, df_volumes, df_survival, cph_model


if __name__ == "__main__":
    patient_objects, df_kinetics, df_volumes, df_survival, cph_model = main()