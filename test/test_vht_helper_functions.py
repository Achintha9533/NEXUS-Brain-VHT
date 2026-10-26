import sys
import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add the parent directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all helper functions and variables from vht_helper_functions
from vht_helper_functions import (
    tp_index,
    safe_num,
    safe_delta,
    get_numeric,
    get_clinical_day,
    treatment_phase,
    resolve_category_value,
    biological_priors,
    treatment_columns,
    active_labels,
    label_map
)


class TestVHTHelperFunctions(unittest.TestCase):
    """Unit tests for NEXUS Brain VHT shared configuration and helper utilities."""

    def test_configurations_exist(self):
        """Verify that core configuration constants and mappings are properly defined."""
        self.assertIsInstance(biological_priors, list)
        self.assertIn("IDH1 mutation", biological_priors)
        self.assertIn("MGMT methylation", biological_priors)

        self.assertIsInstance(treatment_columns, list)
        self.assertIn("Radiation Therapy", treatment_columns)

        self.assertIsInstance(active_labels, list)
        self.assertEqual(len(active_labels), 4)

        self.assertIsInstance(label_map, dict)
        self.assertIn("Enhancing Tumor Core (Label3)", label_map)

    def test_tp_index(self):
        """Test extraction of numeric indices from timepoint strings."""
        self.assertEqual(tp_index("Timepoint_1"), 1)
        self.assertEqual(tp_index("TP_03"), 3)
        self.assertEqual(tp_index("10"), 10)
        self.assertEqual(tp_index("NoDigits"), 10**9)
        self.assertEqual(tp_index(None), 10**9)

    def test_safe_num(self):
        """Test safe float parsing from mixed types and strings with units."""
        self.assertEqual(safe_num(42), 42.0)
        self.assertEqual(safe_num(3.14), 3.14)
        self.assertEqual(safe_num("60 Gy"), 60.0)
        self.assertEqual(safe_num("30 fractions"), 30.0)
        self.assertEqual(safe_num(" -12.5 days "), -12.5)
        self.assertTrue(np.isnan(safe_num(np.nan)))
        self.assertTrue(np.isnan(safe_num("NoNumbers")))

    def test_safe_delta(self):
        """Test rate of change calculation handling zero/negative intervals and NaNs."""
        self.assertEqual(safe_delta(10.0, 5.0, 2.0), 2.5)
        self.assertEqual(safe_delta(5.0, 10.0, 2.0), -2.5)
        self.assertTrue(np.isnan(safe_delta(10.0, 5.0, 0.0)))
        self.assertTrue(np.isnan(safe_delta(10.0, 5.0, -1.0)))
        self.assertTrue(np.isnan(safe_delta(np.nan, 5.0, 2.0)))
        self.assertTrue(np.isnan(safe_delta(10.0, np.nan, 2.0)))
        self.assertTrue(np.isnan(safe_delta(10.0, 5.0, np.nan)))

    def test_get_numeric(self):
        """Test extraction and cleaning of numeric clinical metrics from dictionaries."""
        clinical = {
            "Dose": "60 Gy",
            "Number of Fractions": 30,
            "NA_Field": "N/A",
            "Empty_Field": "",
            "None_Field": "None"
        }
        self.assertEqual(get_numeric(clinical, "Dose"), 60.0)
        self.assertEqual(get_numeric(clinical, " Number of Fractions "), 30.0)
        self.assertTrue(np.isnan(get_numeric(clinical, "NA_Field")))
        self.assertTrue(np.isnan(get_numeric(clinical, "Empty_Field")))
        self.assertTrue(np.isnan(get_numeric(clinical, "None_Field")))
        self.assertTrue(np.isnan(get_numeric(clinical, "NonExistentKey")))

    def test_get_clinical_day(self):
        """Test mapping of timepoint strings to absolute clinical days from diagnosis."""
        clinical = {
            "Days_to_Timepoint_1": 0.0,
            "Number of Days from Diagnosis to Timepoint_2": 45.0,
            "Invalid_Key": "NotADay"
        }
        self.assertEqual(get_clinical_day(clinical, "Timepoint_1"), 0.0)
        self.assertEqual(get_clinical_day(clinical, "Timepoint_2"), 45.0)
        self.assertTrue(np.isnan(get_clinical_day(clinical, "Timepoint_99")))
        self.assertTrue(np.isnan(get_clinical_day(clinical, "NoNum")))

    def test_treatment_phase(self):
        """Test categorization of scan days relative to treatment start/end windows."""
        self.assertEqual(treatment_phase(10, 20, 40), "pre")
        self.assertEqual(treatment_phase(25, 20, 40), "active")
        self.assertEqual(treatment_phase(50, 20, 40), "post")
        self.assertEqual(treatment_phase(25, 20, np.nan), "active")
        self.assertEqual(treatment_phase(np.nan, 20, 40), "none")
        self.assertEqual(treatment_phase(10, np.nan, 40), "none")

    def test_resolve_category_value(self):
        """Test dictionary key substring matching for tumor sub-regions."""
        region_dict = {
            "Necrotic Tumor Core (Label1)": 15.0,
            "Tumor Infiltration and Edema": 40.0,
            "Enhancing Tumor Core (Label3)": 25.0,
            "Resection Cavity (Label4)": 5.0
        }
        self.assertEqual(resolve_category_value(region_dict, "Necrotic"), 15.0)
        self.assertEqual(resolve_category_value(region_dict, "Edema"), 40.0)
        self.assertEqual(resolve_category_value(region_dict, "Enhancing"), 25.0)
        self.assertEqual(resolve_category_value(region_dict, "Resection"), 5.0)
        self.assertTrue(np.isnan(resolve_category_value(region_dict, "NonExistent")))
        self.assertTrue(np.isnan(resolve_category_value(None, "Edema")))


if __name__ == "__main__":
    unittest.main()