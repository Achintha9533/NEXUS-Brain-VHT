"""VHT digital-patient pipeline.

Loads longitudinal segmentation/radiomic sequences and clinical data, builds the
per-patient objects, and extracts the kinetic and exposure dataframes.

Run as a script:
    python vht_pipeline.py                      # uses the default data paths below
    python vht_pipeline.py --out-dir outputs    # also saves df_kinetics.csv / df_vol.csv

Import from a notebook or another script:
    from vht_pipeline import *
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from vht_helper_functions import *


# =====================================================================
# 3. DATA LOADING & PATIENT OBJECTS
# =====================================================================

def load_segmentation_sequences(seg_vol_path):
    """Reads every sheet of the segmentation workbook into per-label, per-patient lists."""
    print(" Extracting longitudinal radiomic sequences...")
    m_dict = {k: {} for k in ['vol', 'vox', 't1cm', 't1cs', 't1nm', 't1ns',
                              't2fm', 't2fs', 't2wm', 't2ws']}

    seg_file = pd.ExcelFile(seg_vol_path)
    for sheet in seg_file.sheet_names:
        df = pd.read_excel(seg_vol_path, sheet_name=sheet)
        df.columns = df.columns.str.strip()

        if 'Patient ID' not in df.columns:
            continue

        df['PID_Clean'] = df['Patient ID'].astype(str).str.split('-').str[0]
        label_key = label_map.get(sheet, sheet)
        group = df.groupby('PID_Clean')

        cols = list(df.columns)
        if len(cols) < 13:
            continue

        m_dict['vox'][label_key]  = group[cols[3]].apply(list).to_dict()
        m_dict['vol'][label_key]  = group[cols[4]].apply(list).to_dict()
        m_dict['t1cm'][label_key] = group[cols[5]].apply(list).to_dict()
        m_dict['t1cs'][label_key] = group[cols[6]].apply(list).to_dict()
        m_dict['t1nm'][label_key] = group[cols[7]].apply(list).to_dict()
        m_dict['t1ns'][label_key] = group[cols[8]].apply(list).to_dict()
        m_dict['t2fm'][label_key] = group[cols[9]].apply(list).to_dict()
        m_dict['t2fs'][label_key] = group[cols[10]].apply(list).to_dict()
        m_dict['t2wm'][label_key] = group[cols[11]].apply(list).to_dict()
        m_dict['t2ws'][label_key] = group[cols[12]].apply(list).to_dict()

    return m_dict


def load_clinical_data(clinical_path, sheet_name=CLINICAL_SHEET):
    """Returns (clinical_df, clinical_data) where clinical_data maps PID_Clean -> row dict."""
    clinical_df = pd.read_excel(clinical_path, sheet_name=sheet_name)
    clinical_df['PID_Clean'] = clinical_df['Patient ID'].astype(str).str.split('-').str[0]
    clinical_data = clinical_df.set_index('PID_Clean').to_dict(orient='index')
    return clinical_df, clinical_data


def _get_val(m_dict, key, l_name, clean_id, i):
    seq = m_dict[key].get(l_name, {}).get(clean_id, [])
    return seq[i] if i < len(seq) else 0


def build_patient_objects(image_path, m_dict, clinical_data):
    """Builds the integrated patient objects (images, volumes, intensities, clinical)."""
    image_path = Path(image_path)
    patient_ids = sorted([d for d in os.listdir(image_path)
                          if (image_path / d).is_dir() and not d.startswith('.')])
    patient_objects = {}

    print(" Building integrated VHT Patient objects...")

    for pid in patient_ids:
        clean_id = pid.split('-')[0]
        p_folder = image_path / pid
        tps = sorted([t.name for t in p_folder.iterdir()
                      if t.is_dir() and not t.name.startswith('.')])

        obj = {
            'Patient_ID': pid,
            'PID_Clean': clean_id,
            'clinical': clinical_data.get(clean_id, {}),
            'timepoints': tps,
            'images': {}, 'volumes': {}, 'voxels': {},
            't1c_mean': {}, 't1c_stdevs': {}, 't1n_mean': {}, 't1n_stdevs': {},
            't2f_mean': {}, 't2f_stdevs': {}, 't2w_mean': {}, 't2w_stdevs': {}
        }

        for i, tp in enumerate(tps):
            obj['images'][tp] = [str(f) for f in (p_folder / tp).glob("*.nii*")]
            for l_name in m_dict['vol'].keys():
                def get_val(key):
                    return _get_val(m_dict, key, l_name, clean_id, i)

                obj['volumes'].setdefault(tp, {})[l_name] = float(get_val('vol'))
                obj['voxels'].setdefault(tp, {})[l_name] = int(get_val('vox'))
                obj['t1c_mean'].setdefault(tp, {})[l_name] = float(get_val('t1cm'))
                obj['t1c_stdevs'].setdefault(tp, {})[l_name] = float(get_val('t1cs'))
                obj['t1n_mean'].setdefault(tp, {})[l_name] = float(get_val('t1nm'))
                obj['t1n_stdevs'].setdefault(tp, {})[l_name] = float(get_val('t1ns'))
                obj['t2f_mean'].setdefault(tp, {})[l_name] = float(get_val('t2fm'))
                obj['t2f_stdevs'].setdefault(tp, {})[l_name] = float(get_val('t2fs'))
                obj['t2w_mean'].setdefault(tp, {})[l_name] = float(get_val('t2wm'))
                obj['t2w_stdevs'].setdefault(tp, {})[l_name] = float(get_val('t2ws'))

        patient_objects[pid] = obj

    return patient_objects


# =====================================================================
# 4. FEATURE EXTRACTION & DATA PIPELINE
# =====================================================================

def build_kinetic_features(patient_objects):
    """Extracts longitudinal velocity, intensity, and treatment features (df_kinetics)."""
    kinetic_features = []

    for pid, obj in patient_objects.items():
        clean_id = obj.get('PID_Clean', str(pid).split('-')[0])
        clinical = obj.get('clinical', {})
        tps = sorted(obj.get('timepoints', []), key=tp_index)

        if len(tps) < 2:
            continue

        # --- EXTRACT CLINICAL TREATMENT METADATA ---
        chemo_startday = get_numeric(clinical, 'Number of days from Diagnosis to Initial Chemo Therapy Start date')
        chemo_endday   = get_numeric(clinical, 'Number of days from Diagnosis to Initial Chemo Therapy end date')

        rad_startday   = get_numeric(clinical, 'Number of days from Diagnosis to Radiation Therapy Start date')
        rad_dose       = get_numeric(clinical, 'Dose')
        rad_fractions  = get_numeric(clinical, 'Number of Fractions')
        rad_endday     = get_numeric(clinical, 'Number of days from Diagnosis to Radiation Therapy end date')

        additional1_startday     = get_numeric(clinical, 'Number of Days from Diagnosis to Starting Additional Therapy')
        additional1_cycle_length = get_numeric(clinical, 'Cycle length of Additional Therapy (q days)')
        additional1_cycles       = get_numeric(clinical, 'Number of Cycles of Additional Therapy')
        additional1_endday       = get_numeric(clinical, 'Number of Days from Diagnosis to Complete Additional Therapy')

        additional2_startday     = get_numeric(clinical, 'Number of Days from Diagnosis to Starting 2nd_Additional Therapy')
        additional2_cycle_length = get_numeric(clinical, 'Cycle length of 2nd_Additional Therapy (q days)')
        additional2_cycles       = get_numeric(clinical, 'Number of Cycles of 2nd_Additional Therapy')
        additional2_endday       = get_numeric(clinical, 'Number of Days from Diagnosis to Complete 2nd_Additional Therapy')

        immuno_startday     = get_numeric(clinical, 'Number of Days from Diagnosis to Start Immunotherapy')
        immuno_cycle_length = get_numeric(clinical, 'Cycle length of Immunotherapy (q days)')
        immuno_cycles       = get_numeric(clinical, 'Number of Cycles of Immunotherapy')
        immuno_endday       = get_numeric(clinical, 'Number of Days from Diagnosis to Complete Immunotherapy')

        brachy_startday     = get_numeric(clinical, 'Number of Days from Diagnosis to the day of Insertion of Brachytherapy')

        prev_tp = None
        prev_day = None

        for tp in tps:
            day = get_clinical_day(clinical, tp)
            if pd.isna(day):
                fallback_day = float(tp_index(tp))
                day = fallback_day if fallback_day < 10**9 else (0.0 if prev_day is None else prev_day + 1.0)

            row = {'PID_Clean': clean_id, 'Timepoint': tp, 'Day': float(day)}
            total_volume = 0.0

            for lbl in active_labels:
                v_now = obj.get('volumes', {}).get(tp, {}).get(lbl, np.nan)
                if pd.notna(v_now):
                    total_volume += float(v_now)

                row[f"Volume_{lbl}"] = v_now
                row[f"Vel_{lbl}"] = 0.0
                row[f"VoxelVel_{lbl}"] = 0.0
                row[f"IntensityChange_{lbl}"] = 0.0
                row[f"HeterogeneityChange_{lbl}"] = 0.0

            row["VolumeTotal"] = float(total_volume) if total_volume > 0 else np.nan

            if prev_tp is not None and pd.notna(prev_day):
                dt = day - prev_day
                if dt > 0:
                    for lbl in active_labels:
                        v_now = obj.get('volumes', {}).get(tp, {}).get(lbl, np.nan)
                        v_prev = obj.get('volumes', {}).get(prev_tp, {}).get(lbl, np.nan)
                        vx_now = obj.get('voxels', {}).get(tp, {}).get(lbl, np.nan)
                        vx_prev = obj.get('voxels', {}).get(prev_tp, {}).get(lbl, np.nan)

                        row[f"Vel_{lbl}"] = np.nan_to_num(safe_delta(v_now, v_prev, dt), nan=0.0)
                        row[f"VoxelVel_{lbl}"] = np.nan_to_num(safe_delta(vx_now, vx_prev, dt), nan=0.0)

                        t1c_vel = safe_delta(obj.get('t1c_mean', {}).get(tp, {}).get(lbl, np.nan), obj.get('t1c_mean', {}).get(prev_tp, {}).get(lbl, np.nan), dt)
                        t1n_vel = safe_delta(obj.get('t1n_mean', {}).get(tp, {}).get(lbl, np.nan), obj.get('t1n_mean', {}).get(prev_tp, {}).get(lbl, np.nan), dt)
                        t2f_vel = safe_delta(obj.get('t2f_mean', {}).get(tp, {}).get(lbl, np.nan), obj.get('t2f_mean', {}).get(prev_tp, {}).get(lbl, np.nan), dt)
                        t2w_vel = safe_delta(obj.get('t2w_mean', {}).get(tp, {}).get(lbl, np.nan), obj.get('t2w_mean', {}).get(prev_tp, {}).get(lbl, np.nan), dt)

                        t1cs_vel = safe_delta(obj.get('t1c_stdevs', {}).get(tp, {}).get(lbl, np.nan), obj.get('t1c_stdevs', {}).get(prev_tp, {}).get(lbl, np.nan), dt)
                        t1ns_vel = safe_delta(obj.get('t1n_stdevs', {}).get(tp, {}).get(lbl, np.nan), obj.get('t1n_stdevs', {}).get(prev_tp, {}).get(lbl, np.nan), dt)
                        t2fs_vel = safe_delta(obj.get('t2f_stdevs', {}).get(tp, {}).get(lbl, np.nan), obj.get('t2f_stdevs', {}).get(prev_tp, {}).get(lbl, np.nan), dt)
                        t2ws_vel = safe_delta(obj.get('t2w_stdevs', {}).get(tp, {}).get(lbl, np.nan), obj.get('t2w_stdevs', {}).get(prev_tp, {}).get(lbl, np.nan), dt)

                        w_int = intensity_weights.get(lbl, {'T1C': 0.25, 'T1N': 0.25, 'T2F': 0.25, 'T2W': 0.25})
                        w_het = heterogeneity_weights.get(lbl, {'T1C': 0.25, 'T1N': 0.25, 'T2F': 0.25, 'T2W': 0.25})

                        row[f"IntensityChange_{lbl}"] = (
                            w_int['T1C'] * np.nan_to_num(t1c_vel, nan=0.0) +
                            w_int['T1N'] * np.nan_to_num(t1n_vel, nan=0.0) +
                            w_int['T2F'] * np.nan_to_num(t2f_vel, nan=0.0) +
                            w_int['T2W'] * np.nan_to_num(t2w_vel, nan=0.0)
                        )

                        row[f"HeterogeneityChange_{lbl}"] = (
                            w_het['T1C'] * np.nan_to_num(t1cs_vel, nan=0.0) +
                            w_het['T1N'] * np.nan_to_num(t1ns_vel, nan=0.0) +
                            w_het['T2F'] * np.nan_to_num(t2fs_vel, nan=0.0) +
                            w_het['T2W'] * np.nan_to_num(t2ws_vel, nan=0.0)
                        )

            # --- 1. CATEGORICAL TREATMENT PHASES ---
            row['ChemoPhase']       = treatment_phase(day, chemo_startday, chemo_endday)
            row['RadPhase']         = treatment_phase(day, rad_startday, rad_endday)
            row['Additional1Phase'] = treatment_phase(day, additional1_startday, additional1_endday)
            row['Additional2Phase'] = treatment_phase(day, additional2_startday, additional2_endday)
            row['ImmunoPhase']      = treatment_phase(day, immuno_startday, immuno_endday)
            row['BrachyPhase']      = treatment_phase(day, brachy_startday, brachy_startday)

            # --- 2. RADIOBIOLOGY & DOSIMETRY FEATURES ---
            row['Rad_TotalDose']    = safe_num(rad_dose)
            row['Rad_Fractions']    = safe_num(rad_fractions)
            row['Rad_FractionDose'] = (
                safe_num(rad_dose) / safe_num(rad_fractions)
                if pd.notna(rad_dose) and pd.notna(rad_fractions) and safe_num(rad_fractions) > 0
                else np.nan
            )
            # Biological Effective Dose (BED = D * (1 + d / (alpha/beta))) with alpha/beta = 10 Gy
            if pd.notna(row['Rad_TotalDose']) and pd.notna(row['Rad_FractionDose']):
                row['Rad_BED'] = row['Rad_TotalDose'] * (1.0 + row['Rad_FractionDose'] / 10.0)
            else:
                row['Rad_BED'] = np.nan

            # --- 3. CYCLIC PROTOCOL SPECIFICATIONS ---
            row['Additional1_CycleLength'] = safe_num(additional1_cycle_length)
            row['Additional1_Cycles']      = safe_num(additional1_cycles)

            row['Additional2_CycleLength'] = safe_num(additional2_cycle_length)
            row['Additional2_Cycles']      = safe_num(additional2_cycles)

            row['Immuno_CycleLength']      = safe_num(immuno_cycle_length)
            row['Immuno_Cycles']           = safe_num(immuno_cycles)

            # --- 4. BIOLOGICAL PRIORS & RAW TREATMENT COLS ---
            for marker in biological_priors:
                row[marker] = clinical.get(marker, 0)
            for col in treatment_columns:
                row[col] = clinical.get(col, 0)

            kinetic_features.append(row)
            prev_tp = tp
            prev_day = day

    df_kinetics = pd.DataFrame(kinetic_features)

    # --- POST-PROCESSING & ENCODING ---
    if len(df_kinetics) > 0:
        for col in biological_priors + treatment_columns:
            if col in df_kinetics.columns and df_kinetics[col].dtype == 'object':
                df_kinetics[col] = LabelEncoder().fit_transform(df_kinetics[col].astype(str))

        phase_cols = [c for c in df_kinetics.columns if c.endswith("Phase")]
        if phase_cols:
            df_kinetics = pd.get_dummies(df_kinetics, columns=phase_cols, dummy_na=False)

        for pid, obj in patient_objects.items():
            clean_id = obj.get('PID_Clean', str(pid).split('-')[0])
            patient_sub_df = df_kinetics[df_kinetics['PID_Clean'] == clean_id].copy()
            obj["kinetic_features"] = patient_sub_df if not patient_sub_df.empty else None

    return df_kinetics


def build_patient_volume_dataframe(patient_objects):
    """Converts patient objects into continuous active exposure fractions for PyMC.

    Filters out patients with fewer than 2 longitudinal timepoints.
    """
    rows = []

    for p_idx, (pid, obj) in enumerate(patient_objects.items()):
        tps = obj.get("timepoints", [])
        try:
            tps = sorted(
                tps,
                key=lambda x: (
                    float(x) if str(x).replace(".", "", 1).isdigit() else x
                ),
            )
        except Exception:
            tps = sorted(tps)

        # Skip patients with fewer than 2 timepoints
        if len(tps) < 2:
            continue

        kin_df = obj.get("kinetic_features", None)
        day_mapping = {}
        if (
            isinstance(kin_df, pd.DataFrame)
            and "Timepoint" in kin_df.columns
            and "Day" in kin_df.columns
        ):
            day_mapping = dict(zip(kin_df["Timepoint"], kin_df["Day"]))

        clinical = obj.get("clinical", {}) if isinstance(obj, dict) else {}
        treatment_intervals = {}

        # Use get_numeric to handle strings with units (e.g., '17 days')
        for name, start_key, end_key in keys_to_check:
            s_val = get_numeric(clinical, start_key)
            e_val = get_numeric(clinical, end_key)

            if pd.notna(s_val):
                start_day = max(0.0, float(s_val))
                end_day = float(e_val) if pd.notna(e_val) else float("inf")
                if end_day <= start_day:
                    end_day = start_day + 2.0
                treatment_intervals[name] = (start_day, end_day)

        prev_day = 0.0
        for i, tp in enumerate(tps):
            vols = obj.get("volumes", {}).get(tp, {})
            if not isinstance(vols, dict):
                continue

            v_nec = resolve_category_value(vols, "Necrotic")
            v_ede = resolve_category_value(vols, "Edema")
            v_enh = resolve_category_value(vols, "Enhancing")
            v_res = resolve_category_value(vols, "Resection")

            # Fallback to get_clinical_day if not present in day_mapping
            day = day_mapping.get(tp, None)
            if day is None:
                day = get_clinical_day(clinical, tp)
            if pd.isna(day) or day is None:
                try:
                    day = float(tp_index(tp))
                except Exception:
                    continue

            curr_day = float(day)
            interval_len = curr_day - prev_day if i > 0 else 0.0

            row_data = {
                "Patient_ID": pid,
                "Patient_Idx": p_idx,
                "Timepoint": tp,
                "Day": curr_day,
                "Volume_Necrotic": (
                    float(v_nec) if pd.notna(v_nec) and v_nec > 0 else 0.0
                ),
                "Volume_Enhancing": (
                    float(v_enh) if pd.notna(v_enh) and v_enh > 0 else 0.0
                ),
                "Volume_Edema": (
                    float(v_ede) if pd.notna(v_ede) and v_ede > 0 else 0.0
                ),
                "Volume_Resection": (
                    float(v_res) if pd.notna(v_res) and v_res > 0 else 0.0
                ),
                "Is_Baseline": 1 if i == 0 else 0,
            }

            for name in [
                "Chemo",
                "Rad",
                "Additional1",
                "Additional2",
                "Immuno",
                "Brachy",
            ]:
                active_fraction = 0.0
                if name in treatment_intervals:
                    start, end = treatment_intervals[name]
                    # Robust overlap logic: check if treatment window intersects interval [prev_day, curr_day]
                    if interval_len <= 0:
                        if start <= curr_day <= end:
                            active_fraction = 1.0
                    else:
                        overlap_start = max(prev_day, start)
                        overlap_end = min(curr_day, end)
                        overlap = max(0.0, overlap_end - overlap_start)
                        # If the interval is largely encapsulated by treatment, flag active
                        active_fraction = min(1.0, overlap / interval_len)
                        if overlap > 0 and active_fraction < 0.05:
                            active_fraction = 1.0  # Catch edge cases where minor timing offsets truncate fractional overlap
                row_data[f"{name}_Active"] = active_fraction

            rows.append(row_data)
            prev_day = curr_day

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Re-index Patient_Idx to ensure contiguous 0..N-1 indices for PyMC parameters
    id_map = {pid: idx for idx, pid in enumerate(df["Patient_ID"].unique())}
    df["Patient_Idx"] = df["Patient_ID"].map(id_map)

    df["Volume_Total"] = (
        df["Volume_Necrotic"] + df["Volume_Enhancing"] + df["Volume_Edema"]
    )
    df["Volume_Core"] = df["Volume_Necrotic"] + df["Volume_Enhancing"]

    vmax_map = df.groupby("Patient_Idx")["Volume_Total"].max() * 1.25
    df["Vmax_Ref"] = df["Patient_Idx"].map(vmax_map).clip(lower=500000.0)

    df = df.sort_values(by=["Patient_Idx", "Day"]).reset_index(drop=True)
    df["dt"] = df.groupby("Patient_Idx")["Day"].diff().fillna(0.0)
    return df


# =====================================================================
# 5. ENTRY POINT
# =====================================================================

def run_pipeline(image_path=DEFAULT_IMAGE_PATH,
                 seg_vol_path=DEFAULT_SEG_VOL_PATH,
                 clinical_path=DEFAULT_CLINICAL_PATH):
    """Runs the full pipeline and returns (patient_objects, df_kinetics, df_vol, clinical_df)."""
    m_dict = load_segmentation_sequences(seg_vol_path)
    clinical_df, clinical_data = load_clinical_data(clinical_path)
    patient_objects = build_patient_objects(image_path, m_dict, clinical_data)

    # Order matters: build_kinetic_features attaches 'kinetic_features' to each patient,
    # which build_patient_volume_dataframe then uses as its day mapping.
    df_kinetics = build_kinetic_features(patient_objects)
    df_vol = build_patient_volume_dataframe(patient_objects)
    return patient_objects, df_kinetics, df_vol, clinical_df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build VHT patient objects and feature dataframes.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--seg-vol-path", type=Path, default=DEFAULT_SEG_VOL_PATH)
    parser.add_argument("--clinical-path", type=Path, default=DEFAULT_CLINICAL_PATH)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="If given, save df_kinetics.csv and df_vol.csv here.")
    args = parser.parse_args(argv)

    for label, p in [("image path", args.image_path),
                     ("segmentation workbook", args.seg_vol_path),
                     ("clinical workbook", args.clinical_path)]:
        if not p.exists():
            parser.error(f"{label} not found: {p}")

    patient_objects, df_kinetics, df_vol, clinical_df = run_pipeline(
        args.image_path, args.seg_vol_path, args.clinical_path)

    print(f"\nPatients (image folders): {len(patient_objects)}")
    print(f"Clinical rows:            {len(clinical_df)}")
    print(f"df_kinetics:              {df_kinetics.shape} "
          f"({df_kinetics['PID_Clean'].nunique() if len(df_kinetics) else 0} patients with >=2 scans)")
    print(f"df_vol:                   {df_vol.shape} "
          f"({df_vol['Patient_ID'].nunique() if len(df_vol) else 0} patients)")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        df_kinetics.to_csv(args.out_dir / "df_kinetics.csv", index=False)
        df_vol.to_csv(args.out_dir / "df_vol.csv", index=False)
        print(f"Saved CSVs to {args.out_dir}")

    return patient_objects, df_kinetics, df_vol


if __name__ == "__main__":
    main()