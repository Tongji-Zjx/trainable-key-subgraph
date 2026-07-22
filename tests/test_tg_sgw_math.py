from __future__ import absolute_import, division, print_function

import unittest

import torch

from keysubgraph.theory import (
    DifferentiableGWLoss,
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
    gw_identity_coupling_upper_bound,
    laplacian_fidelity_metrics,
)


class TGSGWMathTest(unittest.TestCase):
    def setUp(self):
        self.full = torch.tensor(
            [[0.0, 0.7, -0.3], [0.7, 0.0, 0.2], [-0.3, 0.2, 0.0]],
            dtype=torch.float64,
        )
        self.mask = self.full.abs() > 0.0
        self.mask.fill_diagonal_(False)
        self.laplacian = SignedLaplacianBuilder(eta=1.0e-3)

    def test_regularized_signed_laplacian_contains_eta_in_numerator(self):
        isolated = torch.zeros(1, 1, dtype=torch.float64)
        matrix = self.laplacian(isolated, edge_mask=torch.zeros(1, 1, dtype=torch.bool))
        self.assertTrue(torch.allclose(matrix, torch.ones_like(matrix)))
        matrix = self.laplacian(self.full, edge_mask=self.mask)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix).min()), -1.0e-10)

    def test_normalized_frobenius_and_operator_diagnostics_have_correct_scale(self):
        soft = self.full * 0.8
        full_l = self.laplacian(self.full, edge_mask=self.mask)
        soft_l = self.laplacian(soft, edge_mask=self.mask)
        result = laplacian_fidelity_metrics(full_l, soft_l)
        reconstructed = 3.0 * torch.sqrt(result.normalized_frobenius_squared)
        self.assertTrue(torch.allclose(reconstructed, result.frobenius_norm))
        self.assertLessEqual(float(result.operator_norm), float(result.frobenius_norm) + 1.0e-12)

    def test_identity_gw_upper_bound_is_differentiable_and_named_separately(self):
        scale = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        full_l = self.laplacian(self.full, edge_mask=self.mask)
        soft_l = self.laplacian(self.full * scale, edge_mask=self.mask)
        heat = HeatKernelMetricBuilder(0.5)
        full_distance = heat(full_l).distance
        soft_distance = heat(soft_l).distance
        upper = gw_identity_coupling_upper_bound(full_distance, soft_distance)
        self.assertGreaterEqual(float(upper.squared_upper_bound), 0.0)
        upper.squared_upper_bound.backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(bool(torch.isfinite(scale.grad)))

    def test_gw_result_distinguishes_structural_and_regularized_values(self):
        full_l = self.laplacian(self.full, edge_mask=self.mask)
        changed_l = self.laplacian(self.full * 0.75, edge_mask=self.mask)
        heat = HeatKernelMetricBuilder(0.5)
        result = DifferentiableGWLoss(max_iter=5, sinkhorn_iter=10)(
            heat(full_l).distance, heat(changed_l).distance
        )
        self.assertIsNotNone(result.regularized_objective)
        self.assertTrue(result.returned_distance_excludes_entropy_term)
        self.assertTrue(torch.allclose(result.structural_cost_sqrt, result.distance))


if __name__ == "__main__":
    unittest.main()
