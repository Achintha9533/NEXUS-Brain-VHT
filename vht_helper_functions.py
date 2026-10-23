# =====================================================================
# vht_helpers.py
# Shared configuration and unified helper functions for NEXUS Brain VHT
# =====================================================================

import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

DATA_ROOT = Path(
    "/Users/kasunachinthaperera/Documents/"
    "Final Year Project:Thesis/Data/"
    "PKG - MU-Glioma-Post"
)
DEFAULT_IMAGE_PATH = DATA_ROOT / "MU-Glioma-Post"
DEFAULT_SEG_VOL_PATH = DATA_ROOT / "MU-Glioma-Post_Segmentation_Volumes.xlsx"
DEFAULT_CLINICAL_PATH = DATA_ROOT / "MU-Glioma-Post_ClinicalData-July2025.xlsx"
CLINICAL_SHEET = "MU Glioma Post"

active_labels = ["Necrotic", "Enhancing", "Edema", "Resection"]

label_map = {
    "Necrotic Tumor Core (Label1)": "Necrotic",
    "Tumor Infiltration and Edema": "Edema",
    "Enhancing Tumor Core (Label3)": "Enhancing",
    "Resection Cavity (Label4)": "Resection",
}

biological_priors = [
    'IDH1 mutation', 'IDH2 mutation', 'MGMT methylation', '1p/19q',
    'ATRX mutation', 'BRAF V600E mutation', 'TERT promoter mutation',
    'Chromosome 7 gain and Chromosome 10 loss', 'H3-3A mutation',
    'EGFR amplification', 'PTEN mutation', 'CDKN2A/B deletion',
    'TP53 alteration', 'Grade of Primary Brain Tumor'
]

treatment_columns = [
    'Initial Chemo Therapy', 'Radiation Therapy', 'Dose', 'Number of Fractions',
    'Additional Therapy', '2nd_Additional Therapy', 'Immuno therapy', 'Brachy therapy',
    'Cycle length of Additional Therapy (q days)',
    'Number of Days from Diagnosis to Starting Additional Therapy',
    'Number of Days from Diagnosis to Starting 2nd_Additional Therapy',
    'Number of Days from Diagnosis to Start Immunotherapy',
    'Number of days from Diagnosis to Initial Chemo Therapy Start date',
    'Number of days from Diagnosis to Radiation Therapy Start date'
]

keys_to_check = [
    ("Chemo", "Number of days from Diagnosis to Initial Chemo Therapy Start date", "Number of days from Diagnosis to Initial Chemo Therapy end date"),
    ("Rad", "Number of days from Diagnosis to Radiation Therapy Start date", "Number of days from Diagnosis to Radiation Therapy end date"),
    ("Additional1", "Number of Days from Diagnosis to Starting Additional Therapy", "Number of Days from Diagnosis to Complete Additional Therapy"),
    ("Additional2", "Number of Days from Diagnosis to Starting 2nd_Additional Therapy", "Number of Days from Diagnosis to Complete 2nd_Additional Therapy"),
    ("Immuno", "Number of Days from Diagnosis to Start Immunotherapy", "Number of Days from Diagnosis to Complete Immunotherapy"),
    ("Brachy", "Number of Days from Diagnosis to the day of Insertion of Brachytherapy", "Number of Days from Diagnosis to the day of Insertion of Brachytherapy")
]

intensity_weights = {
    'Necrotic':  {'T1C': 0.10, 'T1N': 0.30, 'T2F': 0.30, 'T2W': 0.30},
    'Edema':     {'T1C': 0.10, 'T1N': 0.15, 'T2F': 0.40, 'T2W': 0.35},
    'Enhancing': {'T1C': 0.50, 'T1N': 0.15, 'T2F': 0.20, 'T2W': 0.15},
    'Resection': {'T1C': 0.35, 'T1N': 0.25, 'T2F': 0.20, 'T2W': 0.20},
}

heterogeneity_weights = {
    'Necrotic':  {'T1C': 0.15, 'T1N': 0.25, 'T2F': 0.30, 'T2W': 0.30},
    'Edema':     {'T1C': 0.10, 'T1N': 0.15, 'T2F': 0.40, 'T2W': 0.35},
    'Enhancing': {'T1C': 0.45, 'T1N': 0.20, 'T2F': 0.20, 'T2W': 0.15},
    'Resection': {'T1C': 0.30, 'T1N': 0.25, 'T2F': 0.20, 'T2W': 0.25},
}


# =====================================================================
# 2. UNIFIED HELPER FUNCTIONS
# =====================================================================

def tp_index(tp):
    """Extracts numeric index from timepoint string."""
    digits = ''.join(filter(str.isdigit, str(tp)))
    return int(digits) if digits else 10**9


def safe_num(x):
    """Safely parses float values from mixed numeric/string types."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", x)
        if match:
            return float(match.group())
    return np.nan


def safe_delta(now, prev, dt):
    """Calculates rate of change (delta / dt) handling NaNs and zero dt."""
    if pd.isna(now) or pd.isna(prev) or pd.isna(dt) or dt <= 0:
        return np.nan
    return (float(now) - float(prev)) / float(dt)


def get_numeric(clinical, key):
    """
    Extracts numeric values from clinical metadata dictionaries,
    automatically stripping text units like '60 Gy', '30 fractions', 'days', etc.
    """
    clean_key = str(key).strip()
    val = None

    if clean_key in clinical:
        val = clinical.get(clean_key)
    else:
        for k in clinical.keys():
            if str(k).strip() == clean_key:
                val = clinical.get(k)
                break

    if pd.isna(val) or val is None or str(val).strip().upper() in ["NA", "N/A", "NONE", ""]:
        return np.nan

    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    if isinstance(val, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", val)
        if match:
            return float(match.group())

    return np.nan


def get_clinical_day(clinical, tp):
    """Maps timepoint ID to clinical days from diagnosis."""
    tp_num = ''.join(filter(str.isdigit, str(tp)))
    if not tp_num:
        return np.nan
    candidates = [k for k in clinical.keys() if f"Timepoint_{tp_num}" in str(k)]
    if not candidates:
        return np.nan
    val = clinical.get(candidates[0], np.nan)
    try:
        return float(val)
    except Exception:
        return np.nan


def treatment_phase(day, start, end):
    """Categorizes timepoint into treatment window phase."""
    if pd.isna(day) or pd.isna(start):
        return "none"
    if day < start:
        return "pre"
    if pd.notna(end) and start <= day <= end:
        return "active"
    if pd.notna(end) and day > end:
        return "post"
    return "active"


def resolve_category_value(region_dict, category):
    """Resolves sub-region volume keys via substring matching."""
    if not isinstance(region_dict, dict):
        return np.nan
    category_lower = category.lower()
    for k, v in region_dict.items():
        k_lower = str(k).lower()
        if category_lower == 'edema' and ('edema' in k_lower or 'infiltration' in k_lower):
            return v
        if category_lower in k_lower:
            return v
    return np.nan