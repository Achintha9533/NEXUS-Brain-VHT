# =====================================================================
# test_cox_survival_ht.py
#
# Pytest suite for cox_survival_ht.py
#
# Folder structure:
#   NEXUS Brain VHT/
#   ├── cox_survival_ht.py
#   └── test/
#       └── test_cox_survival_ht.py
#
# Run from NEXUS Brain VHT/:
#   pytest -v test/test_cox_survival_ht.py
#
# Tests:
#   - Helper functions
#   - build_kinetic_features()
#   - build_patient_volume_dataframe()
#   - build_survival_dataset()
#   - run_cox_survival_analysis()
#   - Patient-object mutation
#   - Treatment exposure fractions
#   - Volume and dt calculations
# =====================================================================

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the project root (NEXUS Brain VHT/) to Python's module search path.
# This is required because the test file is inside test/, while
# cox_survival_ht.py is in the parent directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cox_survival_ht as dpo


ACTIVE_LABELS = ["Necrotic", "Enhancing", "Edema", "Resection"]


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def synthetic_patient():
    """
    Create one synthetic patient with known timepoints, volumes,
    intensities, survival records, and treatment intervals.
    """
    clinical = {
        "Overall Survival (Days)": 500,
        "Vital Status": "Deceased",
        "Age at Diagnosis": 58,
        "Number of Days from Diagnosis to Initial Chemo Therapy Start date": 10,
        "Number of days from Diagnosis to Initial Chemo Therapy end date": 40,
        "Number of days from Diagnosis to Radiation Therapy Start date": 20,
        "Number of days from Diagnosis to Radiation Therapy end date": 50,
        "Dose": 60,
        "Number of Fractions": 30,
        "IDH1 mutation": 1,
        "IDH2 mutation": 0,
        "MGMT methylation": 1,
        "1p/19q": 0,
        "ATRX mutation": 0,
        "BRAF V600E mutation": 0,
        "TERT promoter mutation": 0,
        "Chromosome 7 gain and Chromosome 10 loss": 0,
        "H3-3A mutation": 0,
        "EGFR amplification": 0,
        "PTEN mutation": 0,
        "CDKN2A/B deletion": 0,
        "TP53 alteration": 0,
        "Grade of Primary Brain Tumor": 4,
        "Initial Chemo Therapy": 1,
        "Radiation Therapy": 1,
        "Additional Therapy": 0,
        "2nd_Additional Therapy": 0,
        "Immuno therapy": 0,
        "Brachy therapy": 0,
    }

    timepoints = [
        "Timepoint_1",
        "Timepoint_2",
        "Timepoint_3",
    ]

    volumes = {
        "Timepoint_1": {
            "Necrotic": 100.0,
            "Enhancing": 200.0,
            "Edema": 300.0,
            "Resection": 0.0,
        },
        "Timepoint_2": {
            "Necrotic": 120.0,
            "Enhancing": 260.0,
            "Edema": 280.0,
            "Resection": 0.0,
        },
        "Timepoint_3": {
            "Necrotic": 150.0,
            "Enhancing": 320.0,
            "Edema": 250.0,
            "Resection": 0.0,
        },
    }

    voxels = {
        "Timepoint_1": {
            "Necrotic": 10,
            "Enhancing": 20,
            "Edema": 30,
            "Resection": 0,
        },
        "Timepoint_2": {
            "Necrotic": 12,
            "Enhancing": 26,
            "Edema": 28,
            "Resection": 0,
        },
        "Timepoint_3": {
            "Necrotic": 15,
            "Enhancing": 32,
            "Edema": 25,
            "Resection": 0,
        },
    }

    def constant_sequence(value):
        return {
            timepoint: {
                label: value
                for label in ACTIVE_LABELS
            }
            for timepoint in timepoints
        }

    obj = {
        "Patient_ID": "SYNTH-001",
        "PID_Clean": "SYNTH",
        "clinical": clinical,
        "timepoints": timepoints,
        "images": {
            timepoint: [
                "scan_{}.nii.gz".format(timepoint)
            ]
            for timepoint in timepoints
        },
        "volumes": volumes,
        "voxels": voxels,
        "t1c_mean": constant_sequence(10.0),
        "t1c_stdevs": constant_sequence(1.0),
        "t1n_mean": constant_sequence(20.0),
        "t1n_stdevs": constant_sequence(2.0),
        "t2f_mean": constant_sequence(30.0),
        "t2f_stdevs": constant_sequence(3.0),
        "t2w_mean": constant_sequence(40.0),
        "t2w_stdevs": constant_sequence(4.0),
    }

    return {"SYNTH-001": obj}


@pytest.fixture
def kinetic_patient(synthetic_patient):
    """
    Return a synthetic patient after kinetic features are built.
    """
    patient_objects = synthetic_patient

    df_kinetics = dpo.build_kinetic_features(patient_objects)

    return patient_objects, df_kinetics


# ---------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------

def test_tp_index():
    assert dpo.tp_index("Timepoint_1") == 1
    assert dpo.tp_index("Timepoint_12") == 12
    assert dpo.tp_index("baseline") == 10 ** 9


def test_safe_num():
    assert dpo.safe_num(10) == 10.0
    assert dpo.safe_num("60 Gy") == 60.0
    assert dpo.safe_num("30 fractions") == 30.0
    assert np.isnan(dpo.safe_num(None))
    assert np.isnan(dpo.safe_num("no number"))


def test_safe_delta():
    assert dpo.safe_delta(120, 100, 20) == 1.0
    assert dpo.safe_delta(100, 120, 20) == -1.0
    assert np.isnan(dpo.safe_delta(np.nan, 100, 20))
    assert np.isnan(dpo.safe_delta(100, np.nan, 20))
    assert np.isnan(dpo.safe_delta(100, 100, 0))


def test_get_numeric_with_units():
    clinical = {
        "Dose": "60 Gy",
        "Number of Fractions": "30 fractions",
        "Missing": "N/A",
    }

    assert dpo.get_numeric(clinical, "Dose") == 60.0
    assert dpo.get_numeric(clinical, "Number of Fractions") == 30.0
    assert np.isnan(dpo.get_numeric(clinical, "Missing"))
    assert np.isnan(dpo.get_numeric(clinical, "Not Present"))


def test_get_clinical_day():
    clinical = {
        "Number of Days from Diagnosis to 1st MRI (Timepoint_1) ": 10,
        "Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ": 40,
    }

    assert dpo.get_clinical_day(clinical, "Timepoint_1") == 10.0
    assert dpo.get_clinical_day(clinical, "Timepoint_2") == 40.0
    assert np.isnan(dpo.get_clinical_day(clinical, "Timepoint_3"))


def test_treatment_phase():
    assert dpo.treatment_phase(5, 10, 40) == "pre"
    assert dpo.treatment_phase(20, 10, 40) == "active"
    assert dpo.treatment_phase(50, 10, 40) == "post"
    assert dpo.treatment_phase(np.nan, 10, 40) == "none"
    assert dpo.treatment_phase(20, np.nan, 40) == "none"


def test_resolve_category_value():
    volumes = {
        "Necrotic Tumor Core": 100.0,
        "Tumor Infiltration and Edema": 300.0,
        "Enhancing Tumor Core": 200.0,
        "Resection Cavity": 0.0,
    }

    assert dpo.resolve_category_value(volumes, "Necrotic") == 100.0
    assert dpo.resolve_category_value(volumes, "Edema") == 300.0
    assert dpo.resolve_category_value(volumes, "Enhancing") == 200.0
    assert dpo.resolve_category_value(volumes, "Resection") == 0.0
    assert np.isnan(dpo.resolve_category_value({}, "Necrotic"))


# ---------------------------------------------------------------------
# Kinetic-feature tests
# ---------------------------------------------------------------------

def test_build_kinetic_features_returns_dataframe(kinetic_patient):
    _, df_kinetics = kinetic_patient

    assert isinstance(df_kinetics, pd.DataFrame)
    assert not df_kinetics.empty
    assert len(df_kinetics) == 3


def test_kinetic_features_have_required_columns(kinetic_patient):
    _, df_kinetics = kinetic_patient

    required_columns = {
        "PID_Clean",
        "Timepoint",
        "Day",
        "VolumeTotal",
        "Rad_TotalDose",
        "Rad_Fractions",
        "Rad_FractionDose",
        "Rad_BED",
    }

    for label in ACTIVE_LABELS:
        required_columns.update(
            {
                "Volume_{}".format(label),
                "Vel_{}".format(label),
                "VoxelVel_{}".format(label),
                "IntensityChange_{}".format(label),
                "HeterogeneityChange_{}".format(label),
            }
        )

    missing_columns = required_columns - set(df_kinetics.columns)

    assert not missing_columns, (
        "Missing kinetic-feature columns: {}".format(
            sorted(missing_columns)
        )
    )


def test_kinetic_days_are_correct(kinetic_patient):
    _, df_kinetics = kinetic_patient

    assert df_kinetics["Day"].tolist() == [10.0, 40.0, 100.0]


def test_kinetic_volume_total_excludes_resection(kinetic_patient):
    _, df_kinetics = kinetic_patient

    expected_total = [600.0, 660.0, 720.0]

    assert np.allclose(
        df_kinetics["VolumeTotal"].to_numpy(),
        expected_total,
    )


def test_kinetic_velocity_calculations(kinetic_patient):
    _, df_kinetics = kinetic_patient

    # Enhancing:
    # TP2: (260 - 200) / (40 - 10) = 2.0
    # TP3: (320 - 260) / (100 - 40) = 1.0
    assert np.allclose(
        df_kinetics["Vel_Enhancing"].to_numpy(),
        [0.0, 2.0, 1.0],
    )

    # Necrotic:
    # TP2: (120 - 100) / 30 = 0.666666...
    # TP3: (150 - 120) / 60 = 0.5
    assert np.allclose(
        df_kinetics["Vel_Necrotic"].to_numpy(),
        [0.0, 2.0 / 3.0, 0.5],
    )

    # Edema:
    # TP2: (280 - 300) / 30 = -0.666666...
    # TP3: (250 - 280) / 60 = -0.5
    assert np.allclose(
        df_kinetics["Vel_Edema"].to_numpy(),
        [0.0, -2.0 / 3.0, -0.5],
    )


def test_baseline_kinetic_features_are_zero(kinetic_patient):
    _, df_kinetics = kinetic_patient

    first_row = df_kinetics.iloc[0]

    for label in ACTIVE_LABELS:
        assert first_row["Vel_{}".format(label)] == 0.0
        assert first_row["VoxelVel_{}".format(label)] == 0.0
        assert first_row["IntensityChange_{}".format(label)] == 0.0
        assert first_row["HeterogeneityChange_{}".format(label)] == 0.0


def test_radiation_bed_calculation(kinetic_patient):
    _, df_kinetics = kinetic_patient

    # Dose = 60 Gy, fractions = 30
    # Fraction dose = 2 Gy
    # BED = 60 * (1 + 2 / 10) = 72
    assert np.allclose(
        df_kinetics["Rad_BED"].to_numpy(),
        [72.0, 72.0, 72.0],
    )

    assert np.allclose(
        df_kinetics["Rad_FractionDose"].to_numpy(),
        [2.0, 2.0, 2.0],
    )


def test_phase_columns_are_dummy_encoded(kinetic_patient):
    _, df_kinetics = kinetic_patient

    expected_phase_columns = {
        "ChemoPhase_pre",
        "ChemoPhase_active",
        "ChemoPhase_post",
        "RadPhase_pre",
        "RadPhase_active",
        "RadPhase_post",
    }

    missing_columns = expected_phase_columns - set(
        df_kinetics.columns
    )

    assert not missing_columns, (
        "Missing phase dummy columns: {}".format(
            sorted(missing_columns)
        )
    )

    assert df_kinetics["ChemoPhase_pre"].tolist() == [1, 0, 0]
    assert df_kinetics["ChemoPhase_active"].tolist() == [0, 1, 0]
    assert df_kinetics["ChemoPhase_post"].tolist() == [0, 0, 1]

    assert df_kinetics["RadPhase_pre"].tolist() == [1, 0, 0]
    assert df_kinetics["RadPhase_active"].tolist() == [0, 1, 0]
    assert df_kinetics["RadPhase_post"].tolist() == [0, 0, 1]


def test_kinetic_features_are_attached_to_patient_object(kinetic_patient):
    patient_objects, df_kinetics = kinetic_patient

    obj = patient_objects["SYNTH-001"]

    assert isinstance(
        obj["kinetic_features"],
        pd.DataFrame,
    )

    assert len(obj["kinetic_features"]) == len(df_kinetics)
    assert obj["kinetic_features"]["PID_Clean"].unique().tolist() == [
        "SYNTH"
    ]


# ---------------------------------------------------------------------
# Volume/exposure dataframe tests
# ---------------------------------------------------------------------

def test_build_patient_volume_dataframe_returns_dataframe(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    assert isinstance(df_volumes, pd.DataFrame)
    assert not df_volumes.empty
    assert len(df_volumes) == 3


def test_volume_dataframe_has_required_columns(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    required_columns = {
        "Patient_ID",
        "Patient_Idx",
        "Timepoint",
        "Day",
        "Is_Baseline",
        "Volume_Total",
        "Volume_Core",
        "Vmax_Ref",
        "dt",
        "Volume_Necrotic",
        "Volume_Enhancing",
        "Volume_Edema",
        "Volume_Resection",
        "Chemo_Active",
        "Rad_Active",
        "Additional1_Active",
        "Additional2_Active",
        "Immuno_Active",
        "Brachy_Active",
    }

    missing_columns = required_columns - set(df_volumes.columns)

    assert not missing_columns, (
        "Missing volume-dataframe columns: {}".format(
            sorted(missing_columns)
        )
    )


def test_volume_dataframe_days_and_dt(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    assert df_volumes["Day"].tolist() == [10.0, 40.0, 100.0]

    assert np.allclose(
        df_volumes["dt"].to_numpy(),
        [0.0, 30.0, 60.0],
    )


def test_volume_total_and_core(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    assert np.allclose(
        df_volumes["Volume_Total"].to_numpy(),
        [600.0, 660.0, 720.0],
    )

    assert np.allclose(
        df_volumes["Volume_Core"].to_numpy(),
        [300.0, 380.0, 470.0],
    )


def test_vmax_reference(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    # Maximum Volume_Total = 720
    # Vmax_Ref = 720 * 1.25 = 900
    assert np.allclose(
        df_volumes["Vmax_Ref"].to_numpy(),
        [900.0, 900.0, 900.0],
    )


def test_baseline_flag(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    assert df_volumes["Is_Baseline"].tolist() == [1, 0, 0]


def test_treatment_exposure_fractions(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    # Chemo interval: day 10 to 40.
    assert np.allclose(
        df_volumes["Chemo_Active"].to_numpy(),
        [1.0, 1.0, 0.0],
    )

    # Radiation interval: day 20 to 50.
    # TP2 interval: day 10 to 40 -> overlap 20 / 30 = 0.666666...
    # TP3 interval: day 40 to 100 -> overlap 10 / 60 = 0.166666...
    assert np.allclose(
        df_volumes["Rad_Active"].to_numpy(),
        [0.0, 20.0 / 30.0, 10.0 / 60.0],
    )

    # No additional, second additional, immunotherapy, or brachytherapy.
    for treatment in [
        "Additional1",
        "Additional2",
        "Immuno",
        "Brachy",
    ]:
        column = "{}_Active".format(treatment)

        assert np.allclose(
            df_volumes[column].to_numpy(),
            [0.0, 0.0, 0.0],
        )


def test_treatment_fractions_are_between_zero_and_one(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    treatment_columns = [
        "Chemo_Active",
        "Rad_Active",
        "Additional1_Active",
        "Additional2_Active",
        "Immuno_Active",
        "Brachy_Active",
    ]

    for column in treatment_columns:
        assert df_volumes[column].between(0, 1).all()


def test_patient_index_is_contiguous(kinetic_patient):
    patient_objects, _ = kinetic_patient

    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    assert df_volumes["Patient_Idx"].unique().tolist() == [0]


# ---------------------------------------------------------------------
# Survival Dataset & Cox Model Tests
# ---------------------------------------------------------------------

def test_build_survival_dataset(kinetic_patient):
    patient_objects, df_kinetics = kinetic_patient
    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    df_survival = dpo.build_survival_dataset(patient_objects, df_volumes, df_kinetics)

    assert isinstance(df_survival, pd.DataFrame)
    assert not df_survival.empty
    assert "Duration" in df_survival.columns
    assert "Event" in df_survival.columns
    assert "Baseline_Volume_Total" in df_survival.columns
    assert df_survival.iloc[0]["Duration"] == 500.0
    assert df_survival.iloc[0]["Event"] == 1


def test_run_cox_survival_analysis(kinetic_patient):
    patient_objects, df_kinetics = kinetic_patient
    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)
    df_survival = dpo.build_survival_dataset(patient_objects, df_volumes, df_kinetics)

    # With a single synthetic patient, standard Cox models with multiple continuous covariates 
    # may fail convergence or complain about degrees of freedom, but we can verify execution handling.
    cph_model = dpo.run_cox_survival_analysis(df_survival)
    # If lifelines is present, ensure it handles single-patient or multi-patient gracefully
    assert cph_model is None or hasattr(cph_model, "print_summary")


# ---------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------

def test_patient_with_one_timepoint_is_excluded():
    patient_objects = {
        "SINGLE-001": {
            "Patient_ID": "SINGLE-001",
            "PID_Clean": "SINGLE",
            "clinical": {},
            "timepoints": ["Timepoint_1"],
            "images": {},
            "volumes": {
                "Timepoint_1": {
                    "Necrotic": 100.0,
                    "Enhancing": 200.0,
                    "Edema": 300.0,
                    "Resection": 0.0,
                }
            },
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
    }

    df_kinetics = dpo.build_kinetic_features(patient_objects)
    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    assert df_kinetics.empty
    assert df_volumes.empty


def test_empty_patient_objects_return_empty_dataframes():
    patient_objects = {}

    df_kinetics = dpo.build_kinetic_features(patient_objects)
    df_volumes = dpo.build_patient_volume_dataframe(patient_objects)

    assert isinstance(df_kinetics, pd.DataFrame)
    assert isinstance(df_volumes, pd.DataFrame)
    assert df_kinetics.empty
    assert df_volumes.empty


def test_negative_scan_day_is_detected(kinetic_patient):
    """
    This test documents a known data-quality risk.

    The current pipeline may retain negative Day values if they are
    present in clinical data. This test intentionally fails if negative
    days appear, so they can be cleaned before modeling.
    """
    patient_objects, df_kinetics = kinetic_patient

    invalid_days = df_kinetics[
        df_kinetics["Day"].isna()
        | ~np.isfinite(df_kinetics["Day"])
        | (df_kinetics["Day"] < 0)
    ]

    assert invalid_days.empty, (
        "Negative or invalid Day values found:\n{}".format(
            invalid_days
        )
    )