"""
Grassmannian Manifold Analysis for IRED Matrix Problems

This module uses Geomstats to analyze the geometry of matrix solutions
on the Grassmann manifold Gr(n, k).
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt

try:
    import geomstats.backend as gs
    from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric
    GEOMSTATS_AVAILABLE = True
except ImportError:
    GEOMSTATS_AVAILABLE = False
    print("Warning: Geomstats not installed. Run: pip install geomstats")


@dataclass
class GeometricAnalysisResult:
    """Container for Grassmannian geometry analysis"""
    grassmann_distance: float
    principal_angles: np.ndarray
    energy_value: float
    frobenius_distance: float
    problem_id: str = ""


class GrassmannianAnalyzer:
    """
    Analyze matrix solutions using Grassmannian manifold geometry.

    For matrix completion with rank r, solutions live on Gr(n, r).
    """

    def __init__(self, n: int, k: int, backend: str = "numpy"):
        """
        Args:
            n: Ambient dimension (matrix size)
            k: Subspace dimension (rank)
            backend: "numpy", "pytorch", or "tensorflow"
        """
        if not GEOMSTATS_AVAILABLE:
            raise ImportError("Geomstats required. Install: pip install geomstats")

        self.n = n
        self.k = k

        # Set backend
        if backend == "pytorch":
            gs.set_default_backend("pytorch")
        elif backend == "numpy":
            gs.set_default_backend("numpy")

        # Initialize Grassmannian
        self.grassmann = Grassmannian(n=n, k=k)
        self.metric = self.grassmann.metric

    def matrix_to_grassmann_point(self, matrix: np.ndarray) -> np.ndarray:
        """
        Convert matrix to point on Grassmann manifold.

        Uses SVD to extract k-dimensional column space.

        Args:
            matrix: [n, m] matrix

        Returns:
            Projection matrix [n, n] representing k-dimensional subspace
        """
        # Compute SVD
        U, S, Vt = np.linalg.svd(matrix, full_matrices=True)

        # Take top k singular vectors
        U_k = U[:, :self.k]

        # Create projection matrix: P = U_k @ U_k.T
        # This satisfies P² = P and P.T = P
        P = U_k @ U_k.T

        # Project onto Grassmannian (ensure it's a valid point)
        P = self.grassmann.projection(P)

        return P

    def compute_grassmann_distance(
        self,
        matrix1: np.ndarray,
        matrix2: np.ndarray
    ) -> float:
        """
        Compute Riemannian distance on Grassmann manifold between two matrices.

        Args:
            matrix1: [n, m] matrix
            matrix2: [n, m] matrix

        Returns:
            Riemannian distance d(matrix1, matrix2) on Gr(n, k)
        """
        # Convert to Grassmann points
        P1 = self.matrix_to_grassmann_point(matrix1)
        P2 = self.matrix_to_grassmann_point(matrix2)

        # Compute distance
        dist = self.metric.dist(P1, P2)

        return float(dist)

    def compute_principal_angles(
        self,
        matrix1: np.ndarray,
        matrix2: np.ndarray
    ) -> np.ndarray:
        """
        Compute principal angles between subspaces.

        Principal angles θ_i ∈ [0, π/2] measure how much two subspaces differ.

        Args:
            matrix1: [n, m] matrix
            matrix2: [n, m] matrix

        Returns:
            Principal angles [k,] in ascending order
        """
        # Extract column spaces
        U1, _, _ = np.linalg.svd(matrix1, full_matrices=False)
        U2, _, _ = np.linalg.svd(matrix2, full_matrices=False)

        U1 = U1[:, :self.k]
        U2 = U2[:, :self.k]

        # Compute singular values of U1.T @ U2
        # These are cosines of principal angles
        _, cos_angles, _ = np.linalg.svd(U1.T @ U2)

        # Clip to [0, 1] for numerical stability
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        # Convert to angles
        principal_angles = np.arccos(cos_angles)

        return principal_angles

    def geodesic_interpolation(
        self,
        matrix1: np.ndarray,
        matrix2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Compute geodesic interpolation on Grassmannian.

        Returns point at parameter t ∈ [0, 1] along geodesic from matrix1 to matrix2.

        Args:
            matrix1: Starting matrix [n, m]
            matrix2: Ending matrix [n, m]
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated projection matrix [n, n]
        """
        P1 = self.matrix_to_grassmann_point(matrix1)
        P2 = self.matrix_to_grassmann_point(matrix2)

        # Compute geodesic
        geodesic_fn = self.metric.geodesic(initial_point=P1, end_point=P2)

        # Evaluate at t
        P_t = geodesic_fn(t)

        return P_t

    def parallel_transport(
        self,
        tangent_vec: np.ndarray,
        base_point: np.ndarray,
        end_point: np.ndarray
    ) -> np.ndarray:
        """
        Parallel transport tangent vector along geodesic.

        Args:
            tangent_vec: Tangent vector at base_point
            base_point: Starting point on Grassmannian
            end_point: Ending point on Grassmannian

        Returns:
            Transported tangent vector at end_point
        """
        transported = self.metric.parallel_transport(
            tangent_vec=tangent_vec,
            base_point=base_point,
            end_point=end_point
        )

        return transported

    def decompose_gradient_tangent_normal(
        self,
        gradient: np.ndarray,
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose gradient into tangent and normal components.

        For matrix on Grassmannian, tangent space corresponds to
        rank-preserving perturbations, normal space to rank-increasing.

        Args:
            gradient: Gradient ∇_y E [n, m]
            matrix: Current matrix [n, m]

        Returns:
            (tangent_component, normal_component)
        """
        # Get Grassmann point
        P = self.matrix_to_grassmann_point(matrix)

        # Project gradient onto tangent space at P
        # Tangent space of Gr(n, k) at P consists of matrices of form:
        # T = P @ A + A.T @ P where A is skew-symmetric (n x n)
        # Approximation: tangent ≈ P @ grad + grad.T @ P
        tangent_component = P @ gradient + gradient.T @ P

        # Normal component = total - tangent
        normal_component = gradient - tangent_component

        return tangent_component, normal_component

    def analyze_energy_vs_distance(
        self,
        energy_fn,  # Callable: matrix -> energy
        ground_truth: np.ndarray,
        perturbations: List[np.ndarray]
    ) -> List[GeometricAnalysisResult]:
        """
        Analyze correlation between Grassmannian distance and energy.

        Args:
            energy_fn: Function that computes energy E(matrix)
            ground_truth: Ground truth matrix [n, m]
            perturbations: List of perturbed matrices

        Returns:
            List of GeometricAnalysisResult
        """
        results = []

        for i, perturbed in enumerate(perturbations):
            # Compute geometric metrics
            grassmann_dist = self.compute_grassmann_distance(ground_truth, perturbed)
            principal_angles = self.compute_principal_angles(ground_truth, perturbed)

            # Compute energy
            energy = float(energy_fn(perturbed))

            # Frobenius distance for comparison
            frob_dist = np.linalg.norm(ground_truth - perturbed, ord='fro')

            result = GeometricAnalysisResult(
                grassmann_distance=grassmann_dist,
                principal_angles=principal_angles,
                energy_value=energy,
                frobenius_distance=frob_dist,
                problem_id=f"perturb_{i}"
            )

            results.append(result)

        return results

    def plot_energy_vs_distance(
        self,
        results: List[GeometricAnalysisResult],
        save_path: Optional[str] = None
    ):
        """
        Plot energy vs geometric distance.

        Args:
            results: List of analysis results
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Extract data
        grassmann_dists = [r.grassmann_distance for r in results]
        frob_dists = [r.frobenius_distance for r in results]
        energies = [r.energy_value for r in results]

        # Plot 1: Energy vs Grassmannian distance
        ax = axes[0]
        ax.scatter(grassmann_dists, energies, alpha=0.6)
        ax.set_xlabel("Grassmannian Distance")
        ax.set_ylabel("Energy E(y)")
        ax.set_title("Energy vs Riemannian Distance")
        ax.grid(True, alpha=0.3)

        # Compute correlation
        corr_grass = np.corrcoef(grassmann_dists, energies)[0, 1]
        ax.text(0.05, 0.95, f"Correlation: {corr_grass:.3f}",
                transform=ax.transAxes, verticalalignment='top')

        # Plot 2: Energy vs Frobenius distance (comparison)
        ax = axes[1]
        ax.scatter(frob_dists, energies, alpha=0.6, color='orange')
        ax.set_xlabel("Frobenius Distance")
        ax.set_ylabel("Energy E(y)")
        ax.set_title("Energy vs Frobenius Distance")
        ax.grid(True, alpha=0.3)

        # Compute correlation
        corr_frob = np.corrcoef(frob_dists, energies)[0, 1]
        ax.text(0.05, 0.95, f"Correlation: {corr_frob:.3f}",
                transform=ax.transAxes, verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def compute_curvature_ratio(
        self,
        gradient: np.ndarray,
        matrix: np.ndarray
    ) -> float:
        """
        Compute ratio of normal to tangent gradient components.

        High ratio means energy is steep in normal direction (off-manifold),
        which is expected if energy respects manifold structure.

        Args:
            gradient: ∇_y E [n, m]
            matrix: Current matrix [n, m]

        Returns:
            Ratio ||∇_N E|| / ||∇_T E||
        """
        tangent, normal = self.decompose_gradient_tangent_normal(gradient, matrix)

        norm_tangent = np.linalg.norm(tangent, ord='fro')
        norm_normal = np.linalg.norm(normal, ord='fro')

        if norm_tangent < 1e-10:
            return np.inf

        return norm_normal / norm_tangent


def generate_rank_perturbations(
    matrix: np.ndarray,
    noise_levels: List[float],
    rank_preserving: bool = True
) -> List[np.ndarray]:
    """
    Generate perturbations of a matrix.

    Args:
        matrix: Base matrix [n, m]
        noise_levels: List of noise magnitudes
        rank_preserving: If True, perturb only within column space (tangent)
                        If False, add full-rank noise (normal)

    Returns:
        List of perturbed matrices
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    rank = np.sum(S > 1e-10)

    perturbations = []

    for noise_level in noise_levels:
        if rank_preserving:
            # Perturb within column space (tangent direction)
            # Add noise to U, keeping rank the same
            noise = np.random.randn(*U[:, :rank].shape) * noise_level
            U_perturbed = U[:, :rank] + noise

            # Orthogonalize
            U_perturbed, _ = np.linalg.qr(U_perturbed)

            # Reconstruct
            perturbed = U_perturbed @ np.diag(S[:rank]) @ Vt[:rank, :]
        else:
            # Add full-rank noise (normal direction)
            noise = np.random.randn(*matrix.shape) * noise_level
            perturbed = matrix + noise

        perturbations.append(perturbed)

    return perturbations


if __name__ == "__main__":
    print("Grassmannian Geometry Analysis Module")
    print("Example: Analyze matrix completion on Gr(20, 2)")

    # Example: Grassmann distance between two rank-2 matrices
    if GEOMSTATS_AVAILABLE:
        analyzer = GrassmannianAnalyzer(n=20, k=2)

        # Create two random rank-2 matrices
        np.random.seed(42)
        A = np.random.randn(20, 10)
        U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=False)
        matrix1 = U_A[:, :2] @ np.diag(S_A[:2]) @ Vt_A[:2, :]

        B = np.random.randn(20, 10)
        U_B, S_B, Vt_B = np.linalg.svd(B, full_matrices=False)
        matrix2 = U_B[:, :2] @ np.diag(S_B[:2]) @ Vt_B[:2, :]

        # Compute distance
        dist = analyzer.compute_grassmann_distance(matrix1, matrix2)
        print(f"Grassmannian distance: {dist:.4f}")

        # Compute principal angles
        angles = analyzer.compute_principal_angles(matrix1, matrix2)
        print(f"Principal angles: {angles}")
    else:
        print("Install geomstats to run examples")
