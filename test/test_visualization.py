import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Add the parent directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import from 'visualization'
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

    @patch("visualization.plt.close")
    @patch("visualization.plt.savefig")
    @patch("visualization.plt.suptitle")
    @patch("visualization.az.plot_rank")
    @patch("visualization.az.plot_trace")
    @patch("visualization.az.summary")
    @patch("visualization.pm.sample")
    @patch("visualization.build_hierarchical_fisher_kpp_model")
    @patch("visualization.build_patient_volume_dataframe")
    def test_estimate_and_plot_each_patient_success(
        self, 
        mock_build_vol_df,
        mock_build_model,
        mock_pm_sample, 
        mock_az_summary, 
        mock_plot_trace,
        mock_plot_rank,
        mock_suptitle,
        mock_savefig,
        mock_plt_close
    ):
        """Test successful execution of estimate_and_plot_each_patient with mocked MCMC & plotting."""
        
        # 1. Mock volume dataframe returned by build_patient_volume_dataframe
        mock_build_vol_df.return_value = pd.DataFrame({
            "Patient_ID": ["PATIENT-001", "PATIENT-001"],
            "Patient_Idx": [0, 0],
            "Timepoint": ["Timepoint_1", "Timepoint_2"],
            "Day": [0.0, 30.0],
            "Volume_Total": [50000.0, 52000.0],
            "Volume_Core": [15000.0, 16000.0],
            "Chemo_Active": [0.0, 1.0],
            "Rad_Active": [0.0, 0.0],
            "Additional1_Active": [0.0, 0.0],
            "Additional2_Active": [0.0, 0.0],
            "Immuno_Active": [0.0, 0.0],
            "Brachy_Active": [0.0, 0.0],
            "Vmax_Ref": [500000.0, 500000.0],
            "dt": [30.0, 30.0]
        })

        # 2. Mock model context
        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        # 3. Mock PyMC trace object structure returned by pm.sample()
        mock_trace = MagicMock()
        # pred_volumes mean tensor shape matching dataframe length (2 rows)
        mock_trace.posterior.__getitem__.return_value.mean.return_value.values = np.array([50000.0, 51000.0])
        mock_trace.sample_stats.diverging.values.sum.return_value = 0
        mock_pm_sample.return_value = mock_trace

        # 4. Mock ArviZ az.summary() using a side effect to handle all calls gracefully
        def az_summary_side_effect(*args, **kwargs):
            # Check arguments or call count/signature to return appropriate mock dataframe
            var_names = kwargs.get("var_names", args[1] if len(args) > 1 else None)
            kind = kwargs.get("kind", "mean")
            
            if var_names is not None:
                return pd.DataFrame({
                    "mean": [-4.5, -3.0, 11.5, 0.25],
                    "sd": [0.5, 0.5, 0.2, 0.05],
                    "hdi_3%": [-5.5, -4.0, 11.0, 0.15],
                    "hdi_97%": [-3.5, -2.0, 12.0, 0.35],
                    "r_hat": [1.00, 1.00, 1.01, 1.01],
                    "ess_bulk": [500.0, 500.0, 500.0, 500.0],
                    "ess_tail": [500.0, 500.0, 500.0, 500.0]
                }, index=["mu_log_rho", "mu_log_D", "mu_log_K", "sigma_obs_log_v"])
            else:
                return pd.DataFrame({
                    "mean": [0.01, 0.005, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    "sd": [0.001, 0.001, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    "hdi_3%": [0.008, 0.003, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                    "hdi_97%": [0.012, 0.007, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                    "r_hat": [1.00] * 8,
                    "ess_bulk": [500.0] * 8,
                    "ess_tail": [500.0] * 8
                }, index=[
                    "rho[0]", "D[0]", 
                    "alpha_chemo[0]", "alpha_rad[0]", 
                    "alpha_immuno[0]", "alpha_add1[0]", 
                    "alpha_add2[0]", "alpha_brachy[0]"
                ])

        mock_az_summary.side_effect = az_summary_side_effect

        # 5. Execute function
        result_df = estimate_and_plot_each_patient(
            self.mock_patient_objects, 
            tune_steps=10, 
            draw_steps=10
        )

        # 6. Assertions
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 1)  # 1 patient processed
        self.assertIn("Patient_ID", result_df.columns)
        self.assertIn("rho", result_df.columns)
        self.assertIn("D", result_df.columns)
        self.assertIn("Chemo", result_df.columns)
        
        self.assertEqual(result_df.iloc[0]["Patient_ID"], "PATIENT-001")
        self.assertEqual(result_df.iloc[0]["rho"], 0.01)
        self.assertEqual(result_df.iloc[0]["D"], 0.005)

        # Verify routines were triggered
        mock_pm_sample.assert_called_once()
        self.assertGreaterEqual(mock_az_summary.call_count, 2)
        mock_plot_trace.assert_called_once()
        mock_plot_rank.assert_called_once()
        mock_savefig.assert_called()

    @patch("visualization.build_patient_volume_dataframe")
    def test_estimate_and_plot_empty_dataframe_handling(self, mock_build_vol_df):
        """Test graceful handling and early exit if input patient data yields an empty volume dataframe."""
        mock_build_vol_df.return_value = pd.DataFrame()

        result_df = estimate_and_plot_each_patient(self.mock_patient_objects)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue(result_df.empty)


if __name__ == "__main__":
    unittest.main()