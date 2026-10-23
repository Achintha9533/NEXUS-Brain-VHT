# Creating the Digital Patient objects
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


# =====================================================================
# 0. CONFIGURATION
# =====================================================================

DATA_ROOT = Path(
    "/Users/kasunachinthaperera/Documents/"
    "Final Year Project:Thesis/Data/"
    "PKG - MU-Glioma-Post"
)

image_path = DATA_ROOT / "MU-Glioma-Post"
seg_vol_path = DATA_ROOT / "MU-Glioma-Post_Segmentation_Volumes.xlsx"
clinical_path = DATA_ROOT / "MU-Glioma-Post_ClinicalData-July2025.xlsx"

CLINICAL_SHEET_NAME = "MU Glioma Post"


# =====================================================================
# 1. LOAD CLINICAL DATA
# =====================================================================

print("Loading clinical data...")

clinical_df = pd.read_excel(
    clinical_path,
    sheet_name=CLINICAL_SHEET_NAME,
)

clinical_df["PID_Clean"] = (
    clinical_df["Patient ID"]
    .astype(str)
    .str.split("-")
    .str[0]
)

clinical_data = clinical_df.set_index(
    "PID_Clean"
).to_dict(orient="index")

print(
    "Loaded {} clinical rows for {} unique patients.".format(
        len(clinical_df),
        clinical_df["PID_Clean"].nunique(),
    )
)


# =====================================================================
# 2. LOAD RADIOMIC MASTER SEQUENCES
# =====================================================================

active_labels = ["Necrotic", "Enhancing", "Edema", "Resection"]

label_map = {
    "Necrotic Tumor Core (Label1)": "Necrotic",
    "Tumor Infiltration and Edema": "Edema",
    "Enhancing Tumor Core (Label3)": "Enhancing",
    "Resection Cavity (Label4)": "Resection",
}


print("Extracting longitudinal radiomic sequences...")


m_dict = {
    key: {}
    for key in [
        "vol", "vox", "t1cm", "t1cs", "t1nm",
        "t1ns", "t2fm", "t2fs", "t2wm", "t2ws",
    ]
}


seg_file = pd.ExcelFile(seg_vol_path)

for sheet in seg_file.sheet_names:
    df = pd.read_excel(
        seg_vol_path,
        sheet_name=sheet,
    )

    df.columns = df.columns.str.strip()

    if "Patient ID" not in df.columns:
        continue

    df["PID_Clean"] = (
        df["Patient ID"]
        .astype(str)
        .str.split("-")
        .str[0]
    )

    label_key = label_map.get(sheet, sheet)
    group = df.groupby("PID_Clean")

    cols = list(df.columns)

    if len(cols) < 13:
        continue

    m_dict["vox"][label_key] = group[cols[3]].apply(list).to_dict()
    m_dict["vol"][label_key] = group[cols[4]].apply(list).to_dict()
    m_dict["t1cm"][label_key] = group[cols[5]].apply(list).to_dict()
    m_dict["t1cs"][label_key] = group[cols[6]].apply(list).to_dict()
    m_dict["t1nm"][label_key] = group[cols[7]].apply(list).to_dict()
    m_dict["t1ns"][label_key] = group[cols[8]].apply(list).to_dict()
    m_dict["t2fm"][label_key] = group[cols[9]].apply(list).to_dict()
    m_dict["t2fs"][label_key] = group[cols[10]].apply(list).to_dict()
    m_dict["t2wm"][label_key] = group[cols[11]].apply(list).to_dict()
    m_dict["t2ws"][label_key] = group[cols[12]].apply(list).to_dict()


patient_ids = sorted(
    [
        directory
        for directory in os.listdir(image_path)
        if (image_path / directory).is_dir()
        and not directory.startswith(".")
    ]
)

patient_objects = {}


print("Building integrated VHT Patient objects...")


for pid in patient_ids:
    clean_id = pid.split("-")[0]
    p_folder = image_path / pid

    tps = sorted(
        [
            timepoint.name
            for timepoint in p_folder.iterdir()
            if timepoint.is_dir()
            and not timepoint.name.startswith(".")
        ]
    )

    obj = {
        "Patient_ID": pid,
        "PID_Clean": clean_id,
        "clinical": clinical_data.get(clean_id, {}),
        "timepoints": tps,
        "images": {},
        "volumes": {},
        "voxels": {},
        "t1c_mean": {},
        "t1c_stdevs": {},
        "t1n_mean": {},
        "t1n_stdevs": {},
        "t2f_mean": {},
        "t2f_stdevs": {},
        "t2w_mean": {},
        "t2w_stdevs": {},
    }

    for index, timepoint in enumerate(tps):
        obj["images"][timepoint] = [
            str(image_file)
            for image_file in (p_folder / timepoint).glob("*.nii*")
        ]

        for label_name in m_dict["vol"].keys():
            def get_value(metric_key):
                sequence = m_dict[metric_key].get(
                    label_name, {}
                ).get(clean_id, [])

                if index < len(sequence):
                    return sequence[index]

                return 0

            obj["volumes"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("vol"))

            obj["voxels"].setdefault(
                timepoint, {}
            )[label_name] = int(get_value("vox"))

            obj["t1c_mean"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t1cm"))

            obj["t1c_stdevs"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t1cs"))

            obj["t1n_mean"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t1nm"))

            obj["t1n_stdevs"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t1ns"))

            obj["t2f_mean"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t2fm"))

            obj["t2f_stdevs"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t2fs"))

            obj["t2w_mean"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t2wm"))

            obj["t2w_stdevs"].setdefault(
                timepoint, {}
            )[label_name] = float(get_value("t2ws"))

    patient_objects[pid] = obj


print(
    "Built {} digital patient objects.".format(
        len(patient_objects)
    )
)


# =====================================================================================
# 3. UNIFIED UTILITIES
# =====================================================================================

biological_priors = [
    "IDH1 mutation",
    "IDH2 mutation",
    "MGMT methylation",
    "1p/19q",
    "ATRX mutation",
    "BRAF V600E mutation",
    "TERT promoter mutation",
    "Chromosome 7 gain and Chromosome 10 loss",
    "H3-3A mutation",
    "EGFR amplification",
    "PTEN mutation",
    "CDKN2A/B deletion",
    "TP53 alteration",
    "Grade of Primary Brain Tumor",
]


treatment_columns = [
    "Initial Chemo Therapy",
    "Radiation Therapy",
    "Dose",
    "Number of Fractions",
    "Additional Therapy",
    "2nd_Additional Therapy",
    "Immuno therapy",
    "Brachy therapy",
    "Cycle length of Additional Therapy (q days)",
    "Number of Days from Diagnosis to Starting Additional Therapy",
    "Number of Days from Diagnosis to Starting 2nd_Additional Therapy",
    "Number of Days from Diagnosis to Start Immunotherapy",
    "Number of days from Diagnosis to Initial Chemo Therapy Start date",
    "Number of days from Diagnosis to Radiation Therapy Start date",
]


keys_to_check = [
    (
        "Chemo",
        "Number of days from Diagnosis to Initial Chemo Therapy Start date",
        "Number of days from Diagnosis to Initial Chemo Therapy end date",
    ),
    (
        "Rad",
        "Number of days from Diagnosis to Radiation Therapy Start date",
        "Number of days from Diagnosis to Radiation Therapy end date",
    ),
    (
        "Additional1",
        "Number of Days from Diagnosis to Starting Additional Therapy",
        "Number of Days from Diagnosis to Complete Additional Therapy",
    ),
    (
        "Additional2",
        "Number of Days from Diagnosis to Starting 2nd_Additional Therapy",
        "Number of Days from Diagnosis to Complete 2nd_Additional Therapy",
    ),
    (
        "Immuno",
        "Number of Days from Diagnosis to Start Immunotherapy",
        "Number of Days from Diagnosis to Complete Immunotherapy",
    ),
    (
        "Brachy",
        "Number of Days from Diagnosis to the day of Insertion of Brachytherapy",
        "Number of Days from Diagnosis to the day of Insertion of Brachytherapy",
    ),
]


intensity_weights = {
    "Necrotic": {
        "T1C": 0.10, "T1N": 0.30, "T2F": 0.30, "T2W": 0.30
    },
    "Edema": {
        "T1C": 0.10, "T1N": 0.15, "T2F": 0.40, "T2W": 0.35
    },
    "Enhancing": {
        "T1C": 0.50, "T1N": 0.15, "T2F": 0.20, "T2W": 0.15
    },
    "Resection": {
        "T1C": 0.35, "T1N": 0.25, "T2F": 0.20, "T2W": 0.20
    },
}


heterogeneity_weights = {
    "Necrotic": {
        "T1C": 0.15, "T1N": 0.25, "T2F": 0.30, "T2W": 0.30
    },
    "Edema": {
        "T1C": 0.10, "T1N": 0.15, "T2F": 0.40, "T2W": 0.35
    },
    "Enhancing": {
        "T1C": 0.45, "T1N": 0.20, "T2F": 0.20, "T2W": 0.15
    },
    "Resection": {
        "T1C": 0.30, "T1N": 0.25, "T2F": 0.20, "T2W": 0.25
    },
}


# =====================================================================
# 4. UNIFIED HELPER FUNCTIONS
# =====================================================================

def tp_index(timepoint):
    """Extracts numeric index from timepoint string."""
    digits = "".join(
        filter(str.isdigit, str(timepoint))
    )

    if digits:
        return int(digits)

    return 10 ** 9


def safe_num(value):
    """Safely parses float values from mixed numeric/string types."""
    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return float(value)

    if isinstance(value, str):
        match = re.search(
            r"[-+]?\d*\.\d+|\d+",
            value,
        )

        if match:
            return float(match.group())

    return np.nan


def safe_delta(current, previous, dt):
    """Calculates rate of change, handling NaNs and zero dt."""
    if (
        pd.isna(current)
        or pd.isna(previous)
        or pd.isna(dt)
        or dt <= 0
    ):
        return np.nan

    return (
        float(current) - float(previous)
    ) / float(dt)


def get_numeric(clinical, key):
    """
    Extracts numeric values from clinical metadata dictionaries,
    automatically stripping text units such as '60 Gy' or '30 fractions'.
    """
    clean_key = str(key).strip()
    value = None

    if clean_key in clinical:
        value = clinical.get(clean_key)
    else:
        for existing_key in clinical.keys():
            if str(existing_key).strip() == clean_key:
                value = clinical.get(existing_key)
                break

    if (
        pd.isna(value)
        or value is None
        or str(value).strip().upper()
        in ["NA", "N/A", "NONE", ""]
    ):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return float(value)

    if isinstance(value, str):
        match = re.search(
            r"[-+]?\d*\.\d+|\d+",
            value,
        )

        if match:
            return float(match.group())

    return np.nan


def get_clinical_day(clinical, timepoint):
    """Maps timepoint ID to clinical days from diagnosis."""
    timepoint_number = "".join(
        filter(str.isdigit, str(timepoint))
    )

    if not timepoint_number:
        return np.nan

    candidates = [
        key for key in clinical.keys()
        if "Timepoint_{}".format(timepoint_number) in str(key)
    ]

    if not candidates:
        return np.nan

    value = clinical.get(candidates[0], np.nan)

    try:
        return float(value)
    except Exception:
        return np.nan


def treatment_phase(day, start, end):
    """Categorizes timepoint into treatment-window phase."""
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

    for key, value in region_dict.items():
        key_lower = str(key).lower()

        if category_lower == "edema" and (
            "edema" in key_lower
            or "infiltration" in key_lower
        ):
            return value

        if category_lower in key_lower:
            return value

    return np.nan


# =====================================================================
# 5. FEATURE EXTRACTION & DATA PIPELINE
# =====================================================================

def build_kinetic_features(patient_objects):
    """Extracts longitudinal velocity, intensity, and treatment features."""
    kinetic_features = []

    for patient_id, obj in patient_objects.items():
        clean_id = obj.get(
            "PID_Clean",
            str(patient_id).split("-")[0],
        )

        clinical = obj.get("clinical", {})

        timepoints = sorted(
            obj.get("timepoints", []),
            key=tp_index,
        )

        if len(timepoints) < 2:
            continue

        chemo_startday = get_numeric(
            clinical,
            "Number of days from Diagnosis to Initial Chemo Therapy Start date",
        )

        chemo_endday = get_numeric(
            clinical,
            "Number of days from Diagnosis to Initial Chemo Therapy end date",
        )

        rad_startday = get_numeric(
            clinical,
            "Number of days from Diagnosis to Radiation Therapy Start date",
        )

        rad_dose = get_numeric(clinical, "Dose")
        rad_fractions = get_numeric(
            clinical,
            "Number of Fractions",
        )

        rad_endday = get_numeric(
            clinical,
            "Number of days from Diagnosis to Radiation Therapy end date",
        )

        additional1_startday = get_numeric(
            clinical,
            "Number of Days from Diagnosis to Starting Additional Therapy",
        )

        additional1_cycle_length = get_numeric(
            clinical,
            "Cycle length of Additional Therapy (q days)",
        )

        additional1_cycles = get_numeric(
            clinical,
            "Number of Cycles of Additional Therapy",
        )

        additional1_endday = get_numeric(
            clinical,
            "Number of Days from Diagnosis to Complete Additional Therapy",
        )

        additional2_startday = get_numeric(
            clinical,
            "Number of Days from Diagnosis to Starting 2nd_Additional Therapy",
        )

        additional2_cycle_length = get_numeric(
            clinical,
            "Cycle length of 2nd_Additional Therapy (q days)",
        )

        additional2_cycles = get_numeric(
            clinical,
            "Number of Cycles of 2nd_Additional Therapy",
        )

        additional2_endday = get_numeric(
            clinical,
            "Number of Days from Diagnosis to Complete 2nd_Additional Therapy",
        )

        immuno_startday = get_numeric(
            clinical,
            "Number of Days from Diagnosis to Start Immunotherapy",
        )

        immuno_cycle_length = get_numeric(
            clinical,
            "Cycle length of Immunotherapy (q days)",
        )

        immuno_cycles = get_numeric(
            clinical,
            "Number of Cycles of Immunotherapy",
        )

        immuno_endday = get_numeric(
            clinical,
            "Number of Days from Diagnosis to Complete Immunotherapy",
        )

        brachy_startday = get_numeric(
            clinical,
            "Number of Days from Diagnosis to the day of Insertion of Brachytherapy",
        )

        previous_timepoint = None
        previous_day = None

        for timepoint in timepoints:
            day = get_clinical_day(
                clinical,
                timepoint,
            )

            if pd.isna(day):
                fallback_day = float(tp_index(timepoint))

                if fallback_day < 10 ** 9:
                    day = fallback_day
                elif previous_day is None:
                    day = 0.0
                else:
                    day = previous_day + 1.0

            row = {
                "PID_Clean": clean_id,
                "Timepoint": timepoint,
                "Day": float(day),
            }

            total_volume = 0.0

            for label in active_labels:
                current_volume = obj.get(
                    "volumes", {}
                ).get(timepoint, {}).get(
                    label, np.nan
                )

                if pd.notna(current_volume):
                    total_volume += float(current_volume)

                row["Volume_{}".format(label)] = current_volume
                row["Vel_{}".format(label)] = 0.0
                row["VoxelVel_{}".format(label)] = 0.0
                row["IntensityChange_{}".format(label)] = 0.0
                row["HeterogeneityChange_{}".format(label)] = 0.0

            row["VolumeTotal"] = (
                float(total_volume)
                if total_volume > 0
                else np.nan
            )

            if (
                previous_timepoint is not None
                and pd.notna(previous_day)
            ):
                dt = day - previous_day

                if dt > 0:
                    for label in active_labels:
                        current_volume = obj.get(
                            "volumes", {}
                        ).get(timepoint, {}).get(
                            label, np.nan
                        )

                        previous_volume = obj.get(
                            "volumes", {}
                        ).get(previous_timepoint, {}).get(
                            label, np.nan
                        )

                        current_voxels = obj.get(
                            "voxels", {}
                        ).get(timepoint, {}).get(
                            label, np.nan
                        )

                        previous_voxels = obj.get(
                            "voxels", {}
                        ).get(previous_timepoint, {}).get(
                            label, np.nan
                        )

                        row["Vel_{}".format(label)] = np.nan_to_num(
                            safe_delta(
                                current_volume,
                                previous_volume,
                                dt,
                            ),
                            nan=0.0,
                        )

                        row["VoxelVel_{}".format(label)] = np.nan_to_num(
                            safe_delta(
                                current_voxels,
                                previous_voxels,
                                dt,
                            ),
                            nan=0.0,
                        )

                        t1c_velocity = safe_delta(
                            obj.get("t1c_mean", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t1c_mean", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        t1n_velocity = safe_delta(
                            obj.get("t1n_mean", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t1n_mean", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        t2f_velocity = safe_delta(
                            obj.get("t2f_mean", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t2f_mean", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        t2w_velocity = safe_delta(
                            obj.get("t2w_mean", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t2w_mean", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        t1c_sd_velocity = safe_delta(
                            obj.get("t1c_stdevs", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t1c_stdevs", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        t1n_sd_velocity = safe_delta(
                            obj.get("t1n_stdevs", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t1n_stdevs", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        t2f_sd_velocity = safe_delta(
                            obj.get("t2f_stdevs", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t2f_stdevs", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        t2w_sd_velocity = safe_delta(
                            obj.get("t2w_stdevs", {}).get(
                                timepoint, {}
                            ).get(label, np.nan),
                            obj.get("t2w_stdevs", {}).get(
                                previous_timepoint, {}
                            ).get(label, np.nan),
                            dt,
                        )

                        intensity_weight = intensity_weights.get(
                            label,
                            {
                                "T1C": 0.25,
                                "T1N": 0.25,
                                "T2F": 0.25,
                                "T2W": 0.25,
                            },
                        )

                        heterogeneity_weight = heterogeneity_weights.get(
                            label,
                            {
                                "T1C": 0.25,
                                "T1N": 0.25,
                                "T2F": 0.25,
                                "T2W": 0.25,
                            },
                        )

                        row[
                            "IntensityChange_{}".format(label)
                        ] = (
                            intensity_weight["T1C"]
                            * np.nan_to_num(t1c_velocity, nan=0.0)
                            + intensity_weight["T1N"]
                            * np.nan_to_num(t1n_velocity, nan=0.0)
                            + intensity_weight["T2F"]
                            * np.nan_to_num(t2f_velocity, nan=0.0)
                            + intensity_weight["T2W"]
                            * np.nan_to_num(t2w_velocity, nan=0.0)
                        )

                        row[
                            "HeterogeneityChange_{}".format(label)
                        ] = (
                            heterogeneity_weight["T1C"]
                            * np.nan_to_num(t1c_sd_velocity, nan=0.0)
                            + heterogeneity_weight["T1N"]
                            * np.nan_to_num(t1n_sd_velocity, nan=0.0)
                            + heterogeneity_weight["T2F"]
                            * np.nan_to_num(t2f_sd_velocity, nan=0.0)
                            + heterogeneity_weight["T2W"]
                            * np.nan_to_num(t2w_sd_velocity, nan=0.0)
                        )

            row["ChemoPhase"] = treatment_phase(
                day,
                chemo_startday,
                chemo_endday,
            )

            row["RadPhase"] = treatment_phase(
                day,
                rad_startday,
                rad_endday,
            )

            row["Additional1Phase"] = treatment_phase(
                day,
                additional1_startday,
                additional1_endday,
            )

            row["Additional2Phase"] = treatment_phase(
                day,
                additional2_startday,
                additional2_endday,
            )

            row["ImmunoPhase"] = treatment_phase(
                day,
                immuno_startday,
                immuno_endday,
            )

            row["BrachyPhase"] = treatment_phase(
                day,
                brachy_startday,
                brachy_startday,
            )

            row["Rad_TotalDose"] = safe_num(rad_dose)
            row["Rad_Fractions"] = safe_num(rad_fractions)

            row["Rad_FractionDose"] = (
                safe_num(rad_dose) / safe_num(rad_fractions)
                if (
                    pd.notna(rad_dose)
                    and pd.notna(rad_fractions)
                    and safe_num(rad_fractions) > 0
                )
                else np.nan
            )

            if (
                pd.notna(row["Rad_TotalDose"])
                and pd.notna(row["Rad_FractionDose"])
            ):
                row["Rad_BED"] = row["Rad_TotalDose"] * (
                    1.0 + row["Rad_FractionDose"] / 10.0
                )
            else:
                row["Rad_BED"] = np.nan

            row["Additional1_CycleLength"] = safe_num(
                additional1_cycle_length
            )

            row["Additional1_Cycles"] = safe_num(
                additional1_cycles
            )

            row["Additional2_CycleLength"] = safe_num(
                additional2_cycle_length
            )

            row["Additional2_Cycles"] = safe_num(
                additional2_cycles
            )

            row["Immuno_CycleLength"] = safe_num(
                immuno_cycle_length
            )

            row["Immuno_Cycles"] = safe_num(
                immuno_cycles
            )

            for marker in biological_priors:
                row[marker] = clinical.get(marker, 0)

            for column in treatment_columns:
                row[column] = clinical.get(column, 0)

            kinetic_features.append(row)

            previous_timepoint = timepoint
            previous_day = day

    df_kinetics = pd.DataFrame(kinetic_features)

    if len(df_kinetics) > 0:
        for column in biological_priors + treatment_columns:
            if (
                column in df_kinetics.columns
                and df_kinetics[column].dtype == "object"
            ):
                df_kinetics[column] = LabelEncoder().fit_transform(
                    df_kinetics[column].astype(str)
                )

        phase_columns = [
            column for column in df_kinetics.columns
            if column.endswith("Phase")
        ]

        if phase_columns:
            df_kinetics = pd.get_dummies(
                df_kinetics,
                columns=phase_columns,
                dummy_na=False,
            )

        for patient_id, obj in patient_objects.items():
            clean_id = obj.get(
                "PID_Clean",
                str(patient_id).split("-")[0],
            )

            patient_subset = df_kinetics[
                df_kinetics["PID_Clean"] == clean_id
            ].copy()

            obj["kinetic_features"] = (
                patient_subset
                if not patient_subset.empty
                else None
            )

    return df_kinetics


def build_patient_volume_dataframe(patient_objects):
    """Converts patient objects into continuous exposure fractions for PyMC."""
    rows = []

    for patient_index, (patient_id, obj) in enumerate(
        patient_objects.items()
    ):
        timepoints = obj.get("timepoints", [])

        try:
            timepoints = sorted(
                timepoints,
                key=lambda value: (
                    float(value)
                    if str(value)
                    .replace(".", "", 1)
                    .isdigit()
                    else value
                ),
            )
        except Exception:
            timepoints = sorted(timepoints)

        if len(timepoints) < 2:
            continue

        kinetic_df = obj.get("kinetic_features", None)

        day_mapping = {}

        if (
            isinstance(kinetic_df, pd.DataFrame)
            and "Timepoint" in kinetic_df.columns
            and "Day" in kinetic_df.columns
        ):
            day_mapping = dict(
                zip(
                    kinetic_df["Timepoint"],
                    kinetic_df["Day"],
                )
            )

        clinical = (
            obj.get("clinical", {})
            if isinstance(obj, dict)
            else {}
        )

        treatment_intervals = {}

        for name, start_key, end_key in keys_to_check:
            start_value = get_numeric(clinical, start_key)
            end_value = get_numeric(clinical, end_key)

            if pd.notna(start_value):
                start_day = max(0.0, float(start_value))

                end_day = (
                    float(end_value)
                    if pd.notna(end_value)
                    else float("inf")
                )

                if end_day <= start_day:
                    end_day = start_day + 2.0

                treatment_intervals[name] = (
                    start_day,
                    end_day,
                )

        previous_day = 0.0

        for index, timepoint in enumerate(timepoints):
            volumes = obj.get(
                "volumes", {}
            ).get(timepoint, {})

            if not isinstance(volumes, dict):
                continue

            necrotic_volume = resolve_category_value(
                volumes,
                "Necrotic",
            )

            edema_volume = resolve_category_value(
                volumes,
                "Edema",
            )

            enhancing_volume = resolve_category_value(
                volumes,
                "Enhancing",
            )

            resection_volume = resolve_category_value(
                volumes,
                "Resection",
            )

            day = day_mapping.get(timepoint, None)

            if day is None:
                day = get_clinical_day(
                    clinical,
                    timepoint,
                )

            if pd.isna(day) or day is None:
                try:
                    day = float(tp_index(timepoint))
                except Exception:
                    continue

            current_day = float(day)

            interval_length = (
                current_day - previous_day
                if index > 0
                else 0.0
            )

            row_data = {
                "Patient_ID": patient_id,
                "Patient_Idx": patient_index,
                "Timepoint": timepoint,
                "Day": current_day,
                "Volume_Necrotic": (
                    float(necrotic_volume)
                    if pd.notna(necrotic_volume)
                    and necrotic_volume > 0
                    else 0.0
                ),
                "Volume_Enhancing": (
                    float(enhancing_volume)
                    if pd.notna(enhancing_volume)
                    and enhancing_volume > 0
                    else 0.0
                ),
                "Volume_Edema": (
                    float(edema_volume)
                    if pd.notna(edema_volume)
                    and edema_volume > 0
                    else 0.0
                ),
                "Volume_Resection": (
                    float(resection_volume)
                    if pd.notna(resection_volume)
                    and resection_volume > 0
                    else 0.0
                ),
                "Is_Baseline": 1 if index == 0 else 0,
            }

            for treatment_name in [
                "Chemo",
                "Rad",
                "Additional1",
                "Additional2",
                "Immuno",
                "Brachy",
            ]:
                active_fraction = 0.0

                if treatment_name in treatment_intervals:
                    start_day, end_day = treatment_intervals[
                        treatment_name
                    ]

                    if interval_length <= 0:
                        if start_day <= current_day <= end_day:
                            active_fraction = 1.0
                    else:
                        overlap_start = max(
                            previous_day,
                            start_day,
                        )

                        overlap_end = min(
                            current_day,
                            end_day,
                        )

                        overlap = max(
                            0.0,
                            overlap_end - overlap_start,
                        )

                        active_fraction = min(
                            1.0,
                            overlap / interval_length,
                        )

                        if (
                            overlap > 0
                            and active_fraction < 0.05
                        ):
                            active_fraction = 1.0

                row_data[
                    "{}_Active".format(treatment_name)
                ] = active_fraction

            rows.append(row_data)

            previous_day = current_day

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    id_map = {
        patient_id: index
        for index, patient_id in enumerate(
            df["Patient_ID"].unique()
        )
    }

    df["Patient_Idx"] = df["Patient_ID"].map(id_map)

    df["Volume_Total"] = (
        df["Volume_Necrotic"]
        + df["Volume_Enhancing"]
        + df["Volume_Edema"]
    )

    df["Volume_Core"] = (
        df["Volume_Necrotic"]
        + df["Volume_Enhancing"]
    )

    vmax_map = (
        df.groupby("Patient_Idx")["Volume_Total"].max()
        * 1.25
    )

    df["Vmax_Ref"] = (
        df["Patient_Idx"]
        .map(vmax_map)
        .clip(lower=500000.0)
    )

    df = df.sort_values(
        by=["Patient_Idx", "Day"]
    ).reset_index(drop=True)

    df["dt"] = (
        df.groupby("Patient_Idx")["Day"]
        .diff()
        .fillna(0.0)
    )

    return df


# =====================================================================
# 6. MAIN
# =====================================================================

def main():
    df_kinetics = build_kinetic_features(patient_objects)
    df_volumes = build_patient_volume_dataframe(patient_objects)

    print("\n=== Pipeline summary ===")
    print("Digital patient objects: {}".format(len(patient_objects)))
    print("Kinetic feature rows: {}".format(len(df_kinetics)))
    print("Volume/exposure rows: {}".format(len(df_volumes)))

    return patient_objects, df_kinetics, df_volumes


if __name__ == "__main__":
    patient_objects, df_kinetics, df_volumes = main()
