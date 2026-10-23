import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from lifelines import CoxPHFitter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ======================================================================================
# 1. SETUP & LONGITUDINAL EXTRACTION
# ======================================================================================


clinical_path = Path("/Users/kasunachinthaperera/Documents/Final Year Project:Thesis/Data/PKG - MU-Glioma-Post/MU-Glioma-Post_ClinicalData-July2025.xlsx")
clinical_df = pd.read_excel(clinical_path, sheet_name='MU Glioma Post')
clinical_df['PID_Clean'] = clinical_df['Patient ID'].astype(str).str.split('-').str[0]
clinical_data = clinical_df.set_index('PID_Clean').to_dict(orient='index')


# ======================================================================================
# 2. MERGING & CLEANING FOR ML
# ======================================================================================
"""ALIVE_PREFIX: This block consolidates the extracted radiomic features and clinical data into a unified dataframe for machine learning."""
df_vht = clinical_df.copy()


targets = ['Overall Survival (Death)', 'Number of days from Diagnosis to death (Days)']
exclude = targets + [
    'Patient ID', 'PID_Clean', 'Hospice', 'Time to First Progression (Days)', 'Progression',
    'Type of 1st Progression', 'Second Progression/Recurrence', 'Type of 2nd Progression',
    'Number of Days from Diagnosis to 1st MRI (Timepoint_1) ',
    'Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ',
    'Number of Days from Diagnosis to 3rd MRI (Timepoint_3) ',
    'Number of Days from Diagnosis to 4th MRI (Timepoint_4) ',
    'Number of Days from Diagnosis to 5th MRI (Timepoint_5) ',
    'Number of Days from Diagnosis to 6th MRI (Timepoint_6) '
]


X_raw = df_vht.drop(columns=[c for c in exclude if c in df_vht.columns]).copy()


for col in X_raw.select_dtypes(include=['object']).columns:
    X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))
X_raw = X_raw.fillna(0)


y_labels = pd.to_numeric(df_vht[targets[0]], errors='coerce').fillna(0).astype(int)
surv_duration = pd.to_numeric(df_vht[targets[1]], errors='coerce').fillna(1).astype(float)
surv_event = y_labels.copy()


valid_rows = np.isfinite(surv_duration.values)
X_raw = X_raw.loc[valid_rows].reset_index(drop=True)
y_labels = y_labels.loc[valid_rows].reset_index(drop=True)
surv_duration = surv_duration.loc[valid_rows].reset_index(drop=True)
surv_event = surv_event.loc[valid_rows].reset_index(drop=True)
df_vht = df_vht.loc[valid_rows].reset_index(drop=True)



# ======================================================================================
# 5. SURVIVAL MODELING (COX PCA)
# ======================================================================================
"""ALIVE_PREFIX: This block performs survival analysis using the Cox Proportional Hazards 
model on the PCA-transformed features, linking patient IDs and computing individual risk scores."""
df_cox_pca = pd.DataFrame(X_raw.values, columns=X_raw.columns)
df_cox_pca['duration'] = surv_duration.values
df_cox_pca['event'] = surv_event.values


cox_df = df_cox_pca.drop(columns=[], errors='ignore').copy()
cox_df = cox_df.select_dtypes(include=[np.number]).copy()
cox_df = cox_df.dropna(subset=['duration', 'event'])
cox_df = cox_df[cox_df['duration'] > 0]


cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_df, duration_col='duration', event_col='event')
cph.print_summary()


# --- NEW: Link Patient IDs and extract Individual Risk Scores ---
# 1. Bring the patient IDs back into this dataframe using the index alignment
df_cox_pca['PID_Clean'] = df_vht['PID_Clean'].values


# 2. Compute individual mathematical risk scores (Partial Hazards: exp(βX))
df_cox_pca['risk_score'] = cph.predict_partial_hazard(cox_df)


# 3. Fallback: If any patient was filtered out by the Cox engine, give them a baseline risk of 1.0
df_cox_pca['risk_score'] = df_cox_pca['risk_score'].fillna(1.0)


# 4. Display the distribution of risk scores across the cohort
plt.figure(figsize=(10, 12))
cph.plot()
plt.title('VHT Survival Analysis: Hazard Ratios per Component', fontsize=14, fontweight='bold')
plt.xlabel('Log Hazard Ratio (Risk < 0 > Protection)', fontsize=12)
plt.axvline(0, color='red', linestyle='--', alpha=0.5)
plt.show()
