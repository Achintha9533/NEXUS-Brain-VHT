"""
Diagnostic Script to Test Cox Survival Dataset for NaNs, Infs, and Zero-Variance
=============================================================================
Run from terminal: python3 test_cox_data_integrity.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path so it can find the model file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Ensure root directory is in path to import your modules
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import vht_pipeline
import cox_survival_ht as csh


def audit_cox_dataset():
    print("=========================================================")
    print("       NEXUS VHT: COX SURVIVAL DATA INTEGRITY AUDIT      ")
    print("=========================================================")

    # 1. Run pipeline to fetch baseline data objects
    print("\n[1/3] Running vht_pipeline...")
    try:
        patient_objects, df_kinetics, df_volumes, clinical_df = vht_pipeline.run_pipeline()
        print(f" ✅ Pipeline loaded successfully! Total Patients: {len(patient_objects)}")
    except Exception as e:
        print(f"❌ Error running vht_pipeline: {e}")
        return

    # 2. Build survival dataset
    print("\n[2/3] Building survival dataset via csh.build_survival_dataset()...")
    df_survival = csh.build_survival_dataset(patient_objects, df_volumes, df_kinetics)
    print(f" ✅ Built survival dataset. Shape: {df_survival.shape}")

    if df_survival.empty:
        print("❌ Survival dataset is completely empty!")
        return

    # 3. Comprehensive Audit for NaNs, Infs, and Zero-Variance
    print("\n[3/3] Auditing data features for NaNs, Infs, and Variance...")
    
    feature_cols = [
        "Duration", "Event", 
        "Baseline_Volume_Total", "Baseline_Volume_Enhancing", 
        "Mean_Enhancing_Velocity", "Age"
    ]
    
    for marker in ["IDH1 mutation", "MGMT methylation", "1p/19q"]:
        if marker in df_survival.columns:
            feature_cols.append(marker)

    audit_df = df_survival[[c for c in feature_cols if c in df_survival.columns]].copy()

    print("\n--- Data Inspection Table ---")
    print(f"Total Rows: {len(audit_df)}")
    
    report_rows = []
    for col in audit_df.columns:
        nan_count = audit_df[col].isna().sum()
        inf_count = np.isinf(audit_df[col].astype(float)).sum() if audit_df[col].dtype in [np.float64, np.float32] else 0
        nunique = audit_df[col].nunique()
        dtype = audit_df[col].dtype
        
        status = "✅ OK"
        if nan_count > 0 or inf_count > 0:
            status = "⚠️ Has NaNs/Infs"
        if nunique <= 1:
            status = "❌ Zero Variance"

        report_rows.append({
            "Column": col,
            "Type": dtype,
            "NaNs": nan_count,
            "Infs": inf_count,
            "Unique Values": nunique,
            "Status": status
        })

    report_table = pd.DataFrame(report_rows)
    print(report_table.to_string(index=False))

    print("\n=========================================================")
    print("              AUDIT COMPLETE & VERIFIED                  ")
    print("=========================================================")

if __name__ == "__main__":
    audit_cox_dataset()