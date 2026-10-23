import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Add the parent directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import from 'visualization' (change if your file is named differently)
from visualization import estimate_and_plot_each_patient


class TestHBMVisualizationPipeline(unittest.TestCase):

    def setUp(self):
        """Set up mock patient objects dictionary for testing."""
        self.mock_patient_objects = {
            "PATIENT-001": {
                "Patient_ID": "PATIENT-001",
                "PID_Clean": "PATIENT001",
                "timepoints": ["Timepoint_1", "Timepoint_2"],
                "clinical": {
                    "Days_to_Timepoint_1": 0.0,
                    "Days_to_Timepoint_2": 30.0,
                    "Number of days from Diagnosis to Initial Chemo Therapy Start date": 10.0,
                    "Number of days from Diagnosis to Initial Chemo Therapy end date": 40.0,
                    "Dose": "60 Gy",
                },
                "volumes": {
                    "Timepoint_1": {"Necrotic": 10.0, "Enhancing": 15.0, "Edema": 20.0, "Resection": 5.0},
                    "Timepoint_2": {"Necrotic": 12.0, "Enhancing": 20.0, "Edema": 18.0, "Resection": 5.0},
                },
            }
        }

    @patch("visualization.plt.show")
    @patch("visualization.az.summary")
    @patch("visualization.pm.sample")
    def test_estimate_and_plot_each_patient_success(
        self, 
        mock_pm_sample, 
        mock_az_summary, 
        mock_plt_show
    ):
        """Test successful execution of estimate_and_plot_each_patient with mocked MCMC & plotting."""
        
        # 1. Mock PyMC trace object structure returned by pm.sample()
        mock_trace = MagicMock()
        mock_trace.posterior.__getitem__.return_value.mean.return_value.values = np.array([1000.0, 1100.0])
        mock_pm_sample.return_value = mock_trace

        # 2. Mock ArviZ az.summary() to return a DataFrame containing expected parameter summary stats
        index_labels = [
            "rho[0]", "D[0]", 
            "alpha_chemo[0]", "alpha_rad[0]", 
            "alpha_immuno[0]", "alpha_add1[0]", 
            "alpha_add2[0]", "alpha_brachy[0]"
        ]
        mock_az_summary.return_value = pd.DataFrame({"mean": [0.01, 0.005, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}, index=index_labels)

        # 3. Execute function
        result_df = estimate_and_plot_each_patient(
            self.mock_patient_objects, 
            tune_steps=10, 
            draw_steps=10
        )

        # 4. Assertions
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 1)  # 1 patient processed
        self.assertIn("Patient_ID", result_df.columns)
        self.assertIn("rho", result_df.columns)
        self.assertIn("D", result_df.columns)
        self.assertIn("Chemo", result_df.columns)
        
        self.assertEqual(result_df.iloc[0]["Patient_ID"], "PATIENT-001")
        self.assertEqual(result_df.iloc[0]["rho"], 0.01)
        self.assertEqual(result_df.iloc[0]["D"], 0.005)

        # Verify that MCMC sampling and plotting routines were triggered
        mock_pm_sample.assert_called_once()
        mock_az_summary.assert_called_once()
        mock_plt_show.assert_called_once()

    @patch("visualization.build_patient_volume_dataframe")
    def test_estimate_and_plot_empty_dataframe_handling(self, mock_build_vol_df):
        """Test graceful handling and early exit if input patient data yields an empty volume dataframe."""
        mock_build_vol_df.return_value = pd.DataFrame()

        result_df = estimate_and_plot_each_patient(self.mock_patient_objects)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue(result_df.empty)


if __name__ == "__main__":
    unittest.main()