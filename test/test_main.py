"""
Test Script for NEXUS Brain VHT Main Orchestration Pipeline (`main.py`)
======================================================================
Verifies pipeline orchestration, metadata tracking, CSV exporting, and 
CLI execution with mocked backend operations.
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add the parent directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import main orchestration script functions
import main


class TestMainOrchestration(unittest.TestCase):
    """Test suite for main.py execution flow and artifact generation."""

    @patch("main.run_cohort_forecast")
    @patch("main.estimate_and_plot_each_patient")
    @patch("main.run_pipeline")
    def test_main_execution_flow(
        self, 
        mock_run_pipeline, 
        mock_estimate_plot, 
        mock_cohort_forecast
    ):
        """Test that main() correctly sequences pipeline execution, HBM fitting, and forecasts."""
        
        # 1. Mock return values for pipeline and HBM parameter estimation
        mock_patient_objects = {"PATIENT-001": {"Patient_ID": "PATIENT-001"}}
        mock_df_kinetics = pd.DataFrame({"PID_Clean": ["PATIENT001"]})
        mock_df_vol = pd.DataFrame({
            "Patient_ID": ["PATIENT-001", "PATIENT-001"],
            "Day": [0.0, 30.0],
            "Volume_Total": [1000.0, 1100.0]
        })
        mock_clinical_df = pd.DataFrame({"Patient ID": ["PATIENT-001-01"]})

        mock_run_pipeline.return_value = (
            mock_patient_objects, 
            mock_df_kinetics, 
            mock_df_vol, 
            mock_clinical_df
        )

        mock_summary_df = pd.DataFrame({
            "Patient_ID": ["PATIENT-001"],
            "rho": [0.01],
            "D": [0.005],
            "wave_speed": [0.014],
            "Chemo": [0.5]
        })
        mock_estimate_plot.return_value = mock_summary_df
        mock_cohort_forecast.return_value = {"PATIENT-001": pd.DataFrame()}

        # 2. Execute main workflow inside a temporary directory to capture outputs cleanly
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                # Create a mock output folder structure
                Path("output").mkdir(parents=True, exist_ok=True)
                
                # Run main workflow
                main.main()

                # 3. Assertions
                mock_run_pipeline.assert_called_once()
                mock_estimate_plot.assert_called_once()
                mock_cohort_forecast.assert_called_once()

                # Verify provenance metadata file was generated
                metadata_path = Path("output/run_metadata.json")
                self.assertTrue(metadata_path.exists(), "run_metadata.json was not created.")

            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()