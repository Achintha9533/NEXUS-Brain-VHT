import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Add the parent directory to Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pipeline functions and utilities
from vht_pipeline import (
    tp_index,
    safe_num,
    safe_delta,
    get_numeric,
    get_clinical_day,
    treatment_phase,
    resolve_category_value,
    load_clinical_data,
    load_segmentation_sequences,
    build_patient_objects,
    build_kinetic_features,
    build_patient_volume_dataframe,
    run_pipeline,
    main,
)


class TestVHTPipelineUtilities(unittest.TestCase):
    """Unit tests for standalone helper functions."""

    def test_tp_index(self):
        self.assertEqual(tp_index("Timepoint_1"), 1)
        self.assertEqual(tp_index("TP_03"), 3)
        self.assertEqual(tp_index("10"), 10)
        self.assertEqual(tp_index("NoDigits"), 10**9)

    def test_safe_num(self):
        self.assertEqual(safe_num(42), 42.0)
        self.assertEqual(safe_num(3.14), 3.14)
        self.assertEqual(safe_num("60 Gy"), 60.0)
        self.assertEqual(safe_num("30 fractions"), 30.0)
        self.assertEqual(safe_num(" -12.5 days "), -12.5)
        self.assertTrue(np.isnan(safe_num(np.nan)))
        self.assertTrue(np.isnan(safe_num("NoNumbers")))

    def test_safe_delta(self):
        self.assertEqual(safe_delta(10.0, 5.0, 2.0), 2.5)
        self.assertEqual(safe_delta(5.0, 10.0, 2.0), -2.5)
        self.assertTrue(np.isnan(safe_delta(10.0, 5.0, 0.0)))
        self.assertTrue(np.isnan(safe_delta(10.0, 5.0, -1.0)))
        self.assertTrue(np.isnan(safe_delta(np.nan, 5.0, 2.0)))
        self.assertTrue(np.isnan(safe_delta(10.0, np.nan, 2.0)))

    def test_get_numeric(self):
        clinical = {
            "Dose": "60 Gy",
            "Number of Fractions": 30,
            "NA_Field": "N/A",
            "Empty_Field": "",
        }
        self.assertEqual(get_numeric(clinical, "Dose"), 60.0)
        self.assertEqual(get_numeric(clinical, " Number of Fractions "), 30.0)
        self.assertTrue(np.isnan(get_numeric(clinical, "NA_Field")))
        self.assertTrue(np.isnan(get_numeric(clinical, "Empty_Field")))
        self.assertTrue(np.isnan(get_numeric(clinical, "NonExistentKey")))

    def test_get_clinical_day(self):
        clinical = {
            "Days_to_Timepoint_1": 0.0,
            "Days_to_Timepoint_2": 45.0,
        }
        self.assertEqual(get_clinical_day(clinical, "Timepoint_1"), 0.0)
        self.assertEqual(get_clinical_day(clinical, "TP2"), 45.0)
        self.assertTrue(np.isnan(get_clinical_day(clinical, "Timepoint_99")))
        self.assertTrue(np.isnan(get_clinical_day(clinical, "InvalidTP")))

    def test_treatment_phase(self):
        self.assertEqual(treatment_phase(10, 20, 40), "pre")
        self.assertEqual(treatment_phase(25, 20, 40), "active")
        self.assertEqual(treatment_phase(50, 20, 40), "post")
        self.assertEqual(treatment_phase(25, 20, np.nan), "active")
        self.assertEqual(treatment_phase(np.nan, 20, 40), "none")
        self.assertEqual(treatment_phase(10, np.nan, 40), "none")

    def test_resolve_category_value(self):
        region_dict = {
            "Necrotic Tumor Core (Label1)": 15.0,
            "Tumor Infiltration and Edema": 40.0,
            "Enhancing Tumor Core (Label3)": 25.0,
        }
        self.assertEqual(resolve_category_value(region_dict, "Necrotic"), 15.0)
        self.assertEqual(resolve_category_value(region_dict, "Edema"), 40.0)
        self.assertEqual(resolve_category_value(region_dict, "Enhancing"), 25.0)
        self.assertTrue(np.isnan(resolve_category_value(region_dict, "Resection")))
        self.assertTrue(np.isnan(resolve_category_value(None, "Edema")))


class TestVHTPipelineCore(unittest.TestCase):
    """Integration and feature extraction tests for the VHT pipeline."""

    def setUp(self):
        """Build mock digital patient objects data structure."""
        self.mock_patient_objects = {
            "PATIENT001-POST": {
                "Patient_ID": "PATIENT001-POST",
                "PID_Clean": "PATIENT001",
                "timepoints": ["Timepoint_1", "Timepoint_2"],
                "clinical": {
                    "Days_to_Timepoint_1": 0.0,
                    "Days_to_Timepoint_2": 30.0,
                    "Number of days from Diagnosis to Initial Chemo Therapy Start date": 10.0,
                    "Number of days from Diagnosis to Initial Chemo Therapy end date": 40.0,
                    "Number of days from Diagnosis to Radiation Therapy Start date": 10.0,
                    "Number of days from Diagnosis to Radiation Therapy end date": 40.0,
                    "Dose": "60 Gy",
                    "Number of Fractions": 30,
                    "IDH1 mutation": "Mutated",
                },
                "volumes": {
                    "Timepoint_1": {
                        "Necrotic": 10.0,
                        "Edema": 20.0,
                        "Enhancing": 15.0,
                        "Resection": 5.0,
                    },
                    "Timepoint_2": {
                        "Necrotic": 12.0,
                        "Edema": 18.0,
                        "Enhancing": 20.0,
                        "Resection": 5.0,
                    },
                },
                "voxels": {
                    "Timepoint_1": {"Necrotic": 1000, "Edema": 2000, "Enhancing": 1500, "Resection": 500},
                    "Timepoint_2": {"Necrotic": 1200, "Edema": 1800, "Enhancing": 2000, "Resection": 500},
                },
                "t1c_mean": {
                    "Timepoint_1": {"Necrotic": 100, "Edema": 150, "Enhancing": 200, "Resection": 50},
                    "Timepoint_2": {"Necrotic": 105, "Edema": 145, "Enhancing": 220, "Resection": 50},
                },
                "t1c_stdevs": {
                    "Timepoint_1": {"Necrotic": 10, "Edema": 15, "Enhancing": 20, "Resection": 5},
                    "Timepoint_2": {"Necrotic": 11, "Edema": 14, "Enhancing": 22, "Resection": 5},
                },
                "t1n_mean": {},
                "t1n_stdevs": {},
                "t2f_mean": {},
                "t2f_stdevs": {},
                "t2w_mean": {},
                "t2w_stdevs": {},
            }
        }

    def test_build_kinetic_features(self):
        """Test extraction of kinetics, volumetric velocities, and treatment phase dummy variables."""
        df_kinetics = build_kinetic_features(self.mock_patient_objects)

        self.assertIsInstance(df_kinetics, pd.DataFrame)
        self.assertEqual(len(df_kinetics), 2)  # 2 timepoints
        self.assertIn("PID_Clean", df_kinetics.columns)
        self.assertIn("VolumeTotal", df_kinetics.columns)

        # Check total volume computation
        self.assertEqual(df_kinetics.iloc[0]["VolumeTotal"], 50.0)
        self.assertEqual(df_kinetics.iloc[1]["VolumeTotal"], 55.0)

        # Check Necrotic volumetric velocity: (12 - 10) / (30 - 0) = 0.06666...
        self.assertAlmostEqual(df_kinetics.iloc[1]["Vel_Necrotic"], 0.0666666, places=5)

        # Verify radiobiology BED computation: 60 * (1 + 2 / 10) = 72.0
        self.assertEqual(df_kinetics.iloc[0]["Rad_BED"], 72.0)

        # Check object mutation (attaches 'kinetic_features' dataframe)
        patient_obj = self.mock_patient_objects["PATIENT001-POST"]
        self.assertIn("kinetic_features", patient_obj)
        self.assertIsInstance(patient_obj["kinetic_features"], pd.DataFrame)

    def test_build_patient_volume_dataframe(self):
        """Test construction of exposure fractions dataframe for PyMC modeling."""
        # Run build_kinetic_features first to populate patient 'kinetic_features'
        build_kinetic_features(self.mock_patient_objects)

        df_vol = build_patient_volume_dataframe(self.mock_patient_objects)

        self.assertIsInstance(df_vol, pd.DataFrame)
        self.assertEqual(len(df_vol), 2)
        self.assertIn("Patient_ID", df_vol.columns)
        self.assertIn("Patient_Idx", df_vol.columns)
        self.assertIn("Volume_Total", df_vol.columns)
        self.assertIn("Volume_Core", df_vol.columns)

        # Check baseline indicator
        self.assertEqual(df_vol.iloc[0]["Is_Baseline"], 1)
        self.assertEqual(df_vol.iloc[1]["Is_Baseline"], 0)

        # Check active exposure fraction calculations
        self.assertIn("Chemo_Active", df_vol.columns)
        self.assertIn("Rad_Active", df_vol.columns)


class TestVHTPipelineIOAndMain(unittest.TestCase):
    """Tests for file I/O functions and command line interface execution."""

    def test_load_clinical_data(self):
        """Test reading clinical excel files using a temporary Excel workbook."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Create dummy clinical dataframe
            df_dummy = pd.DataFrame({
                "Patient ID": ["P001-01", "P002-01"],
                "Age": [55, 62],
                "Dose": ["60 Gy", "54 Gy"],
            })

            with pd.ExcelWriter(tmp_path) as writer:
                df_dummy.to_excel(writer, sheet_name="MU Glioma Post", index=False)

            df_loaded, data_dict = load_clinical_data(tmp_path, sheet_name="MU Glioma Post")

            self.assertEqual(len(df_loaded), 2)
            self.assertIn("P001", data_dict)
            self.assertIn("P002", data_dict)
            self.assertEqual(data_dict["P001"]["Age"], 55)

        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    @patch("vht_pipeline.load_segmentation_sequences")
    @patch("vht_pipeline.load_clinical_data")
    @patch("vht_pipeline.build_patient_objects")
    @patch("vht_pipeline.build_kinetic_features")
    @patch("vht_pipeline.build_patient_volume_dataframe")
    def test_run_pipeline(
        self,
        mock_build_vol,
        mock_build_kin,
        mock_build_obj,
        mock_load_clin,
        mock_load_seg,
    ):
        """Test full pipeline invocation with mocked components."""
        mock_load_seg.return_value = {}
        mock_load_clin.return_value = (pd.DataFrame(), {})
        mock_build_obj.return_value = {}
        mock_build_kin.return_value = pd.DataFrame()
        mock_build_vol.return_value = pd.DataFrame()

        patient_objs, df_kinetics, df_vol, clinical_df = run_pipeline("img_path", "seg_path", "clin_path")

        mock_load_seg.assert_called_once_with("seg_path")
        mock_load_clin.assert_called_once_with("clin_path")
        mock_build_obj.assert_called_once()
        mock_build_kin.assert_called_once()
        mock_build_vol.assert_called_once()

    @patch("vht_pipeline.run_pipeline")
    def test_main_cli(self, mock_run_pipeline):
        """Test main command-line interface execution and directory creation."""
        mock_run_pipeline.return_value = ({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            img_p = tmp_path / "images"
            seg_p = tmp_path / "seg.xlsx"
            clin_p = tmp_path / "clin.xlsx"
            out_p = tmp_path / "output"

            img_p.mkdir()
            seg_p.touch()
            clin_p.touch()

            args = [
                "--image-path", str(img_p),
                "--seg-vol-path", str(seg_p),
                "--clinical-path", str(clin_p),
                "--out-dir", str(out_p),
            ]

            main(args)

            mock_run_pipeline.assert_called_once_with(img_p, seg_p, clin_p)
            self.assertTrue(out_p.exists())


if __name__ == "__main__":
    unittest.main()