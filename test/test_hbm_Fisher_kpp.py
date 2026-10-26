"""
Test Script for NEXUS Brain VHT Hierarchical Bayesian Model (HBM)
=================================================================
Verifies PyMC model graph construction, literature-grounded hyper-priors,
non-centered patient-level parameters, and tensor shape dimensions.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import pymc as pm

# Add parent directory to path so it can find the model file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hbm_Fisher_kpp import build_hierarchical_fisher_kpp_model


class TestHierarchicalFisherKPPModel(unittest.TestCase):
    """Unit test suite for the Fisher-KPP Hierarchical Bayesian Model."""

    def setUp(self):
        """Create mock longitudinal cohort data frame for multi-patient testing."""
        np.random.seed(42)
        patients = ["PATIENT-001", "PATIENT-002"]
        data = []

        for pid in patients:
            # 3 timepoints per patient
            prev_vol = 1500.0
            for tp in range(3):
                is_base = 1.0 if tp == 0 else 0.0
                dt = 0.0 if is_base == 1.0 else 60.0
                if not is_base:
                    prev_vol *= 1.1

                data.append({
                    "Patient_ID": pid,
                    "Timepoint": f"Timepoint_{tp + 1}",
                    "dt": dt,
                    "Is_Baseline": is_base,
                    "Volume_Total": prev_vol,
                    "Chemo_ActiveFraction": 0.5 if tp == 1 else 0.0,
                    "Rad_ActiveFraction": 1.0 if tp == 1 else 0.0,
                    "Immuno_ActiveFraction": 0.0,
                    "Additional1_ActiveFraction": 0.0,
                    "Additional2_ActiveFraction": 0.0,
                    "Brachy_ActiveFraction": 0.0,
                })

        self.cohort_df = pd.DataFrame(data)

    def test_model_type(self):
        """Test that the function successfully returns a PyMC Model instance."""
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        self.assertIsInstance(model, pm.Model)

    def test_hyper_priors_and_deterministics(self):
        """Test that essential variables and parameters exist in the model graph."""
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        
        expected = [
            "mu_log_rho", "mu_log_D", "mu_log_K",
            "rho", "D", "K",
            "alpha_chemo", "alpha_rad",
            "pred_volumes", "likelihood"
        ]

        for name in expected:
            self.assertIn(
                name, 
                model.named_vars, 
                f"Missing variable '{name}' in the PyMC model graph."
            )

    def test_patient_level_shapes(self):
        """Test that patient-level parameter arrays match the number of unique patients."""
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        num_patients = self.cohort_df["Patient_ID"].nunique()

        with model:
            rho_var = model.named_vars["rho"]
            self.assertEqual(rho_var.eval().shape, (num_patients,))

            alpha_rad_var = model.named_vars["alpha_rad"]
            self.assertEqual(alpha_rad_var.eval().shape, (num_patients,))

    def test_prediction_tensor_shape(self):
        """Test that predicted volumes match the total number of rows in the cohort dataset."""
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        
        with model:
            pred_vol_var = model.named_vars["pred_volumes"]
            self.assertEqual(pred_vol_var.eval().shape, (len(self.cohort_df),))


if __name__ == "__main__":
    unittest.main()