"""Unit tests for cafm_eqm module. Run locally via:

    python projects/diff-EqM/experiments/cafm_eqm/test_cafm_eqm.py

CPU-only; no cluster, no real EqM model required. Validates:
- c(γ) matches transport.py exactly at canonical γ values.
- target/interpolate shapes + values.
- PGD mining respects L2 ball, produces L_hard >= L_clean.
- training_step.cafm_v10_gen_step returns all expected diagnostic keys.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cafm_eqm.eqm_target import (
    EQM_INTERP, EQM_LAMBDA, c_gamma, eqm_base_loss, eqm_target,
    interpolate, velocity_raw,
)
from cafm_eqm.training_step import (
    cafm_gen_step, cafm_v10_gen_step, prepare_eqm_inputs,
)
from cafm_eqm.v10_mining import mine_hard_example, project_l2_ball, v10_aux_loss


class TestEqMTarget(unittest.TestCase):
    def test_c_gamma_canonical(self):
        """c(γ) at canonical values matches Wang+Du paper exactly (transport.py)."""
        gammas = torch.tensor([0.0, 0.2, 0.5, 0.8, 0.9, 1.0])
        got = c_gamma(gammas)
        # With interp=0.8, λ=4: c=4 for γ≤0.8, then linear to 0 at γ=1.
        expected = torch.tensor([4.0, 4.0, 4.0, 4.0, 2.0, 0.0])
        torch.testing.assert_close(got, expected)

    def test_c_gamma_matches_transport_formula(self):
        """Independent reimplementation of transport.py:get_ct."""
        def ref(t):
            interp = 0.8
            start = 1.0
            return torch.minimum(
                start - (start - 1) / interp * t,
                1 / (1 - interp) - 1 / (1 - interp) * t,
            ) * 4

        gammas = torch.rand(100)
        torch.testing.assert_close(c_gamma(gammas), ref(gammas))

    def test_interpolate(self):
        x = torch.randn(4, 4, 32, 32)
        eps = torch.randn(4, 4, 32, 32)
        g = torch.tensor([0.0, 0.5, 0.8, 1.0])
        xg = interpolate(x, eps, g)
        # γ=0: xg = x; γ=1: xg = eps
        torch.testing.assert_close(xg[0], x[0])
        torch.testing.assert_close(xg[3], eps[3])

    def test_target_shape(self):
        x = torch.randn(4, 4, 32, 32)
        eps = torch.randn(4, 4, 32, 32)
        g = torch.rand(4)
        t = eqm_target(x, eps, g)
        self.assertEqual(t.shape, x.shape)

    def test_velocity_raw(self):
        x = torch.randn(2, 4, 32, 32)
        eps = torch.randn(2, 4, 32, 32)
        v = velocity_raw(x, eps)
        torch.testing.assert_close(v, eps - x)


class TestV10Mining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B = 4
        self.x_gamma = torch.randn(self.B, 4, 32, 32)
        self.target = torch.randn(self.B, 4, 32, 32)
        self.W = torch.randn(4, 4, requires_grad=True)

    def fwd(self, x):
        return torch.einsum("ij,bjhw->bihw", self.W, x)

    def test_project_l2_within_ball(self):
        delta = torch.randn(self.B, 4, 32, 32) * 5.0
        proj = project_l2_ball(delta, eps_radius=0.3)
        norms = proj.flatten(1).norm(dim=1)
        # All projected back to ≤ 0.3
        self.assertTrue((norms <= 0.3 + 1e-4).all().item())
        # Some should be exactly at boundary (since input was large)
        self.assertTrue((norms > 0.29).all().item())

    def test_mine_K1_respects_ball(self):
        delta = mine_hard_example(
            self.fwd, self.x_gamma, self.target, K=1, eps_radius=0.3, lr=0.05,
        )
        norms = delta.flatten(1).norm(dim=1)
        self.assertTrue((norms <= 0.3 + 1e-4).all().item())

    def test_mine_K3_respects_ball(self):
        delta = mine_hard_example(
            self.fwd, self.x_gamma, self.target, K=3, eps_radius=0.3, lr=0.05,
        )
        norms = delta.flatten(1).norm(dim=1)
        self.assertTrue((norms <= 0.3 + 1e-4).all().item())

    def test_mining_increases_loss(self):
        """L_hard should be >= L_clean (PGD ascent finds harder examples)."""
        pred_clean = self.fwd(self.x_gamma)
        l_clean = eqm_base_loss(pred_clean, self.target)

        delta = mine_hard_example(
            self.fwd, self.x_gamma, self.target, K=1, eps_radius=0.3, lr=0.05,
        )
        pred_hard = self.fwd(self.x_gamma + delta)
        l_hard = eqm_base_loss(pred_hard, self.target)

        # L_hard should be at least L_clean (within ε of equal); typically strictly greater.
        self.assertGreaterEqual(l_hard.item() + 1e-4, l_clean.item())

    def test_v10_aux_loss_diagnostics(self):
        l_hard, diag = v10_aux_loss(
            self.fwd, self.x_gamma, self.target, K=1, eps_radius=0.3, lr=0.05,
        )
        self.assertIn("v10/delta_norm_mean", diag)
        self.assertIn("v10/delta_norm_std", diag)
        self.assertIn("v10/l_hard", diag)


class TestTrainingStep(unittest.TestCase):
    def test_prepare_eqm_inputs(self):
        latents = torch.randn(4, 4, 32, 32)
        labels = torch.randint(0, 1000, (4,))
        inputs = prepare_eqm_inputs(latents, labels)
        expected = {"latents", "labels", "noises", "gamma", "x_gamma", "target"}
        self.assertEqual(set(inputs.keys()), expected)
        self.assertEqual(inputs["x_gamma"].shape, latents.shape)
        self.assertEqual(inputs["target"].shape, latents.shape)
        # γ ∈ [0, 1]
        self.assertTrue((inputs["gamma"] >= 0).all().item())
        self.assertTrue((inputs["gamma"] <= 1).all().item())

    def test_cafm_gen_step_keys(self):
        latents = torch.randn(2, 4, 32, 32)
        labels = torch.randint(0, 1000, (2,))
        inputs = prepare_eqm_inputs(latents, labels)

        class MockGen:
            def __call__(self, x, y, t):
                return x * 0.5

        class MockDisJVP:
            def __call__(self, x, y, t, dx, dt):
                B = x.size(0)
                # When dx is stacked [2, B], return [2B] for chunk; else [B].
                out = torch.zeros(B)
                out_jvp = torch.zeros(2 * B if dx.ndim == 5 else B)
                return out, out_jvp

        losses = cafm_gen_step(MockDisJVP(), MockGen(), inputs)
        self.assertIn("loss/gen_adv", losses)
        self.assertIn("loss/gen_ot", losses)
        self.assertIn("loss/total_gen", losses)

    def test_cafm_v10_gen_step_keys(self):
        latents = torch.randn(2, 4, 32, 32)
        labels = torch.randint(0, 1000, (2,))
        inputs = prepare_eqm_inputs(latents, labels)

        class MockGen:
            def __call__(self, x, y, t):
                return x * 0.5

        class MockDisJVP:
            def __call__(self, x, y, t, dx, dt):
                B = x.size(0)
                out = torch.zeros(B)
                out_jvp = torch.zeros(2 * B if dx.ndim == 5 else B)
                return out, out_jvp

        losses = cafm_v10_gen_step(MockDisJVP(), MockGen(), inputs)
        for k in ("loss/gen_adv", "loss/gen_ot", "loss/v10_hard",
                  "loss/v10_base", "v10/ratio", "v10/delta_norm_mean",
                  "loss/total_gen"):
            self.assertIn(k, losses)

    def test_v10_default_lambda_active(self):
        """With lambda_v10 > 0 and λ small, total_gen != gen_adv."""
        latents = torch.randn(2, 4, 32, 32)
        labels = torch.randint(0, 1000, (2,))
        inputs = prepare_eqm_inputs(latents, labels)

        class MockGen:
            def __call__(self, x, y, t):
                return x * 0.5

        class MockDisJVP:
            def __call__(self, x, y, t, dx, dt):
                B = x.size(0)
                out = torch.zeros(B)
                out_jvp = torch.zeros(2 * B if dx.ndim == 5 else B)
                return out, out_jvp

        l_v10 = cafm_v10_gen_step(MockDisJVP(), MockGen(), inputs, lambda_v10=0.1)
        l_no_v10 = cafm_v10_gen_step(MockDisJVP(), MockGen(), inputs, lambda_v10=0.0)
        # With λ=0, total_gen == gen_adv (+ gen_ot which is 0)
        self.assertAlmostEqual(
            l_no_v10["loss/total_gen"].item(), l_no_v10["loss/gen_adv"].item(),
            places=5,
        )
        # With λ=0.1, total includes v10 contribution
        self.assertGreater(
            l_v10["loss/total_gen"].item(), l_v10["loss/gen_adv"].item(),
        )


if __name__ == "__main__":
    unittest.main()
