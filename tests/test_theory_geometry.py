from __future__ import absolute_import, division, print_function

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from keysubgraph.theory import (  # noqa: E402
    DifferentiableGWLoss,
    HeatKernelMetricBuilder,
    SignedLaplacianBuilder,
    SpectralStateExtractor,
    spectral_winf_dense,
    spectral_winf_exact,
)


class TheoryGeometryTest(unittest.TestCase):
    def setUp(self):
        self.adjacency = torch.tensor(
            [
                [0.0, 0.6, -0.2],
                [0.6, 0.0, 0.4],
                [-0.2, 0.4, 0.0],
            ],
            dtype=torch.float64,
        )
        self.edge_mask = self.adjacency.abs() > 0.0
        self.edge_mask.fill_diagonal_(False)
        self.laplacian = SignedLaplacianBuilder(eta=1.0e-4)

    def test_signed_laplacian_is_symmetric_psd_and_identity_loss_is_zero(self):
        matrix = self.laplacian(self.adjacency, edge_mask=self.edge_mask)
        self.assertTrue(torch.allclose(matrix, matrix.transpose(0, 1), atol=1.0e-10))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix).min()), -1.0e-8)
        self.assertLess(float(torch.linalg.matrix_norm(matrix - matrix)), 1.0e-12)

    def test_padding_does_not_change_laplacian_spectrum_or_heat_metric(self):
        padded = torch.zeros(5, 5, dtype=self.adjacency.dtype)
        padded[:3, :3] = self.adjacency
        node_mask = torch.tensor([True, True, True, False, False])
        padded_edge_mask = padded.abs() > 0.0
        direct = self.laplacian(self.adjacency, edge_mask=self.edge_mask)
        padded_laplacian = self.laplacian(
            padded, node_mask=node_mask, edge_mask=padded_edge_mask
        )
        extractor = SpectralStateExtractor((0.25, 0.5, 0.75))
        direct_state = extractor(direct)
        padded_state = extractor(padded_laplacian, node_mask=node_mask)
        self.assertTrue(torch.allclose(direct_state.eigenvalues, padded_state.eigenvalues))
        heat = HeatKernelMetricBuilder(0.7)
        direct_heat = heat(direct)
        padded_heat = heat(padded_laplacian, node_mask=node_mask)
        self.assertTrue(torch.allclose(direct_heat.distance, padded_heat.distance))

    def test_heat_kernel_diffusion_distance_contract(self):
        matrix = self.laplacian(self.adjacency, edge_mask=self.edge_mask)
        state = HeatKernelMetricBuilder(0.5)(matrix)
        self.assertEqual(tuple(state.kernel.shape), (3, 3))
        self.assertEqual(tuple(state.distance.shape), (3, 3))
        self.assertGreaterEqual(float(state.distance.min()), 0.0)
        self.assertTrue(torch.allclose(state.distance, state.distance.transpose(0, 1), atol=1.0e-10))
        self.assertTrue(torch.allclose(torch.diag(state.distance), torch.zeros(3, dtype=state.distance.dtype)))

    def test_weakening_positive_or_negative_edge_produces_positive_errors(self):
        full_laplacian = self.laplacian(self.adjacency, edge_mask=self.edge_mask)
        full_metric = HeatKernelMetricBuilder(0.5)(full_laplacian).distance
        solver = DifferentiableGWLoss(
            entropic_reg=0.05, max_iter=20, tolerance=1.0e-6, sinkhorn_iter=20
        )
        for edge in ((0, 1), (0, 2)):
            modified = self.adjacency.clone()
            modified[edge[0], edge[1]] *= 0.5
            modified[edge[1], edge[0]] *= 0.5
            self.assertEqual(
                int(torch.sign(modified[edge[0], edge[1]])),
                int(torch.sign(self.adjacency[edge[0], edge[1]])),
            )
            changed_laplacian = self.laplacian(modified, edge_mask=self.edge_mask)
            changed_metric = HeatKernelMetricBuilder(0.5)(changed_laplacian).distance
            self.assertGreater(
                float(torch.linalg.matrix_norm(full_laplacian - changed_laplacian)),
                0.0,
            )
            self.assertGreater(float(solver(full_metric, changed_metric).distance), 0.0)

    def test_exact_spectral_winf_matches_dense_reference(self):
        first = torch.tensor([0.0, 0.3, 1.2], dtype=torch.float64)
        second = torch.tensor([0.1, 0.8], dtype=torch.float64)
        exact = spectral_winf_exact(first, second)
        dense = spectral_winf_dense(first, second, grid_size=10001)
        self.assertAlmostEqual(float(spectral_winf_exact(first, first)), 0.0, places=12)
        self.assertAlmostEqual(float(exact), float(spectral_winf_exact(second, first)), places=12)
        self.assertAlmostEqual(float(exact), float(dense), places=10)

    def test_gw_identity_nonnegative_failure_and_gradient(self):
        full_laplacian = self.laplacian(self.adjacency, edge_mask=self.edge_mask)
        full_distance = HeatKernelMetricBuilder(0.5)(full_laplacian).distance
        solver = DifferentiableGWLoss(
            entropic_reg=0.05, max_iter=20, tolerance=1.0e-6, sinkhorn_iter=20
        )
        identity = solver(full_distance, full_distance)
        self.assertLess(float(identity.distance), 1.0e-10)

        scale = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        modified_adjacency = self.adjacency * scale
        modified_laplacian = self.laplacian(
            modified_adjacency, edge_mask=self.edge_mask
        )
        modified_distance = HeatKernelMetricBuilder(0.5)(modified_laplacian).distance
        result = solver(full_distance, modified_distance)
        self.assertGreaterEqual(float(result.distance), 0.0)
        result.distance.backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(bool(torch.isfinite(scale.grad)))

        incomplete = DifferentiableGWLoss(
            entropic_reg=0.05,
            max_iter=1,
            tolerance=1.0e-20,
            sinkhorn_iter=1,
            failure_strategy="use_last",
        )(full_distance, modified_distance.detach())
        self.assertFalse(incomplete.converged)
        self.assertGreater(float(incomplete.distance), 0.0)
        with self.assertRaises(RuntimeError):
            DifferentiableGWLoss(
                entropic_reg=0.05,
                max_iter=1,
                tolerance=1.0e-20,
                sinkhorn_iter=1,
                failure_strategy="raise",
            )(full_distance, modified_distance.detach())


if __name__ == "__main__":
    unittest.main()
