import os
import sys
import unittest
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hbm_Fisher_kpp import build_hierarchical_fisher_kpp_model


def generate_synthetic_cohort(
    num_patients: int = 12,
    num_timepoints: int = 5,
    dt_days: float = 30.0,
    rho_true: float = 0.015,
    D_true: float = 0.05,
    K_true: float = 100_000.0,
    alpha_chemo_true: float = 0.2,
    alpha_rad_true: float = 0.3,
    obs_log_sigma: float = 0.05,
    include_invasive: bool = False,
    inv_log_sigma: float = 0.05,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generates synthetic longitudinal tumor volumes from known (D, rho)
    using the same discrete Fisher-KPP radial step rule as the model, and
    optionally a diffusion-driven invasive compartment volume.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for p_idx in range(num_patients):
        pid = f"SYN-{p_idx + 1:03d}"
        base_vol = float(rng.uniform(2000.0, 8000.0))
        r_curr = (3.0 * base_vol / (4.0 * np.pi)) ** (1.0 / 3.0)

        base_inv_vol = float(base_vol * rng.uniform(1.5, 2.5))
        r_inv_curr = (3.0 * base_inv_vol / (4.0 * np.pi)) ** (1.0 / 3.0)

        for tp in range(num_timepoints):
            is_base = 1.0 if tp == 0 else 0.0
            dt = 0.0 if tp == 0 else dt_days

            chemo_act = 0.5 if tp >= 2 else 0.0
            rad_act = 1.0 if tp == 2 else 0.0

            if tp > 0:
                v_curr = (4.0 / 3.0) * np.pi * (r_curr ** 3.0)
                logistic_cap = max(1.0 - (v_curr / K_true), 1e-4)
                damping = np.exp(-(alpha_chemo_true * chemo_act + alpha_rad_true * rad_act))
                rho_eff = max(rho_true * logistic_cap * damping, 1e-8)
                v_fisher = 2.0 * np.sqrt(D_true * rho_eff)
                r_curr = r_curr + dt * v_fisher
                clean_vol = (4.0 / 3.0) * np.pi * (r_curr ** 3.0)

                v_inv = np.sqrt(max(D_true * damping, 1e-8))
                r_inv_curr = r_inv_curr + dt * v_inv
                clean_inv_vol = (4.0 / 3.0) * np.pi * (r_inv_curr ** 3.0)
            else:
                clean_vol = base_vol
                clean_inv_vol = base_inv_vol

            noisy_vol = float(clean_vol * np.exp(rng.normal(0.0, obs_log_sigma)))

            row = {
                "Patient_ID": pid,
                "Timepoint": f"T{tp}",
                "dt": dt,
                "Is_Baseline": is_base,
                "Volume_Total": noisy_vol,
                "Chemo_ActiveFraction": chemo_act,
                "Rad_ActiveFraction": rad_act,
                "Immuno_ActiveFraction": 0.0,
                "Additional1_ActiveFraction": 0.0,
                "Additional2_ActiveFraction": 0.0,
                "Brachy_ActiveFraction": 0.0,
            }

            if include_invasive:
                row["Volume_Invasive"] = float(
                    clean_inv_vol * np.exp(rng.normal(0.0, inv_log_sigma))
                )

            rows.append(row)

    return pd.DataFrame(rows)


class TestHierarchicalFisherKPPModel(unittest.TestCase):
    """Unit and identifiability tests for the Fisher-KPP HBM."""

    def setUp(self):
        self.cohort_df = generate_synthetic_cohort(
            num_patients=2,
            num_timepoints=3,
            dt_days=60.0,
            obs_log_sigma=0.02,
            seed=42,
        )

    def test_model_type(self):
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        self.assertIsInstance(model, pm.Model)

    def test_hyper_priors_and_deterministics(self):
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        expected = [
            "mu_log_prod", "mu_log_ratio",
            "mu_log_rho", "mu_log_D", "mu_log_K",
            "rho", "D", "K",
            "alpha_chemo", "alpha_rad",
            "pred_volumes", "likelihood",
        ]
        for name in expected:
            self.assertIn(name, model.named_vars)

    def test_invasive_observable_adds_likelihood(self):
        inv_df = generate_synthetic_cohort(
            num_patients=2,
            num_timepoints=3,
            dt_days=60.0,
            include_invasive=True,
            seed=42,
        )
        model = build_hierarchical_fisher_kpp_model(inv_df)
        for name in ["sigma_obs_log_inv", "pred_invasive_volumes", "invasive_likelihood"]:
            self.assertIn(name, model.named_vars)

    def test_patient_level_shapes(self):
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        num_patients = self.cohort_df["Patient_ID"].nunique()
        with model:
            self.assertEqual(model.named_vars["rho"].eval().shape, (num_patients,))
            self.assertEqual(model.named_vars["alpha_rad"].eval().shape, (num_patients,))

    def test_prediction_tensor_shape(self):
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        with model:
            self.assertEqual(model.named_vars["pred_volumes"].eval().shape, (len(self.cohort_df),))

    def test_mcmc_smoke_sampling_and_diagnostics(self):
        model = build_hierarchical_fisher_kpp_model(self.cohort_df)
        with model:
            trace = pm.sample(
                draws=10,
                tune=10,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False,
            )
            self.assertIsInstance(trace, az.InferenceData)
            self.assertIn("posterior", trace.groups())
            summary = az.summary(
                trace,
                var_names=["mu_log_prod", "mu_log_ratio", "mu_log_rho", "mu_log_D", "mu_log_K"],
            )
            self.assertIn("r_hat", summary.columns)
            self.assertIn("ess_bulk", summary.columns)

    @unittest.skipUnless(
        os.environ.get("RUN_IDENTIFIABILITY") == "1",
        "Set RUN_IDENTIFIABILITY=1 to run synthetic D-vs-rho identifiability experiment.",
    )
    def test_synthetic_identifiability_constant_product(self):
        """
        Tests product recovery and D-vs-rho identifiability when D1 * rho1 == D2 * rho2.
        Set RUN_INVASIVE_IDENTIFIABILITY=1 to test separation with Volume_Invasive included.
        """
        include_inv = os.environ.get("RUN_INVASIVE_IDENTIFIABILITY") == "1"

        # Same product D * rho = 0.0008
        df_case_a = generate_synthetic_cohort(
            num_patients=15,
            num_timepoints=5,
            D_true=0.08,
            rho_true=0.01,
            obs_log_sigma=0.03,
            include_invasive=include_inv,
            seed=101,
        )
        df_case_b = generate_synthetic_cohort(
            num_patients=15,
            num_timepoints=5,
            D_true=0.01,
            rho_true=0.08,
            obs_log_sigma=0.03,
            include_invasive=include_inv,
            seed=102,
        )

        traces = {}
        for label, df_syn in [("CaseA_highD_lowRho", df_case_a), ("CaseB_lowD_highRho", df_case_b)]:
            model = build_hierarchical_fisher_kpp_model(df_syn)
            with model:
                trace = pm.sample(
                    draws=500,
                    tune=500,
                    chains=2,
                    cores=1,
                    target_accept=0.9,
                    random_seed=42,
                    return_inferencedata=True,
                    progressbar=False,
                )
            post = trace.posterior
            log_D_samples = post["mu_log_D"].values.ravel()
            log_rho_samples = post["mu_log_rho"].values.ravel()
            corr_log = np.corrcoef(log_D_samples, log_rho_samples)[0, 1]
            prod_mean = float(np.exp(log_D_samples + log_rho_samples).mean())

            print(
                f"\n[{label} | invasive={include_inv}] "
                f"mean(mu_log_D)={log_D_samples.mean():.3f}, "
                f"mean(mu_log_rho)={log_rho_samples.mean():.3f}, "
                f"corr(mu_log_D, mu_log_rho)={corr_log:.3f}, "
                f"mean(exp(mu_log_D + mu_log_rho))={prod_mean:.6f}"
            )
            traces[label] = trace

        prod_a = float(
            np.exp(
                traces["CaseA_highD_lowRho"].posterior["mu_log_D"]
                + traces["CaseA_highD_lowRho"].posterior["mu_log_rho"]
            ).mean()
        )
        prod_b = float(
            np.exp(
                traces["CaseB_lowD_highRho"].posterior["mu_log_D"]
                + traces["CaseB_lowD_highRho"].posterior["mu_log_rho"]
            ).mean()
        )
        self.assertAlmostEqual(prod_a, prod_b, delta=5e-4)

        if include_inv:
            mean_D_a = float(traces["CaseA_highD_lowRho"].posterior["mu_log_D"].mean())
            mean_D_b = float(traces["CaseB_lowD_highRho"].posterior["mu_log_D"].mean())
            mean_rho_a = float(traces["CaseA_highD_lowRho"].posterior["mu_log_rho"].mean())
            mean_rho_b = float(traces["CaseB_lowD_highRho"].posterior["mu_log_rho"].mean())
            self.assertGreater(mean_D_a, mean_D_b + 0.5)
            self.assertLess(mean_rho_a, mean_rho_b - 0.5)


if __name__ == "__main__":
    unittest.main()