"""
Hessian Eigenspectrum Analysis for IRED Energy Landscapes

This module computes Hessian matrices H = ∇²_y E^k(x, y*) at correct solutions
and analyzes the eigenspectrum to understand energy landscape geometry.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class HessianAnalysisResult:
    """Container for Hessian analysis results"""
    eigenvalues: np.ndarray  # Shape: (n_eigs,)
    eigenvectors: Optional[np.ndarray] = None  # Shape: (dim, n_eigs)
    condition_number: float = 0.0
    effective_rank: float = 0.0
    spectral_gap: float = 0.0
    annealing_level: int = 0
    problem_id: str = ""


class HessianAnalyzer:
    """
    Compute and analyze Hessian of IRED energy function.

    Uses Lanczos iteration for efficient eigenvalue computation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        n_eigenvalues: int = 20,
        use_lanczos: bool = True
    ):
        """
        Args:
            model: IRED energy model E_θ
            device: "cuda" or "cpu"
            n_eigenvalues: Number of top eigenvalues to compute
            use_lanczos: Use Lanczos iteration (efficient for large matrices)
        """
        self.model = model.to(device)
        self.device = device
        self.n_eigenvalues = n_eigenvalues
        self.use_lanczos = use_lanczos

    def compute_hessian_eigenspectrum(
        self,
        x: torch.Tensor,  # Input condition
        y: torch.Tensor,  # Solution (requires_grad=True)
        t: torch.Tensor,  # Annealing level
        compute_eigenvectors: bool = False
    ) -> HessianAnalysisResult:
        """
        Compute Hessian eigenspectrum at point (x, y) for annealing level t.

        Args:
            x: Input tensor [batch, inp_dim]
            y: Solution tensor [batch, out_dim], requires_grad=True
            t: Annealing level [batch]
            compute_eigenvectors: Whether to compute eigenvectors (expensive)

        Returns:
            HessianAnalysisResult with eigenvalues and analysis metrics
        """
        assert y.requires_grad, "y must have requires_grad=True"

        if self.use_lanczos:
            return self._compute_lanczos_eigenvalues(x, y, t, compute_eigenvectors)
        else:
            return self._compute_full_hessian_eigenvalues(x, y, t, compute_eigenvectors)

    def _compute_lanczos_eigenvalues(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        compute_eigenvectors: bool
    ) -> HessianAnalysisResult:
        """
        Compute top eigenvalues using Lanczos iteration.

        This is memory-efficient for large matrices.
        """
        try:
            from pyhessian import hessian as pyhessian_hessian
        except ImportError:
            raise ImportError("PyHessian not installed. Run: pip install pyhessian")

        # Create energy function wrapper
        def energy_fn():
            # Concatenate input and output for EBM
            inp = torch.cat([x, y], dim=-1)
            energy = self.model(inp, t)
            return energy.sum()  # Must be scalar

        # Compute Hessian
        hessian_comp = pyhessian_hessian(
            self.model,
            energy_fn,
            cuda=(self.device == "cuda")
        )

        # Get top eigenvalues
        eigenvalues, eigenvectors = hessian_comp.eigenvalues(
            top_n=self.n_eigenvalues,
            return_eigenvector=compute_eigenvectors
        )

        eigenvalues = np.array(eigenvalues)

        # Compute analysis metrics
        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 1e-10 else np.inf
        effective_rank = self._compute_effective_rank(eigenvalues)
        spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0.0

        result = HessianAnalysisResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors if compute_eigenvectors else None,
            condition_number=condition_number,
            effective_rank=effective_rank,
            spectral_gap=spectral_gap,
            annealing_level=int(t.item()),
            problem_id=""
        )

        return result

    def _compute_full_hessian_eigenvalues(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        compute_eigenvectors: bool
    ) -> HessianAnalysisResult:
        """
        Compute full Hessian matrix (expensive, only for small problems).
        """
        # Concatenate input and output
        inp = torch.cat([x, y], dim=-1)

        # Compute energy
        energy = self.model(inp, t).sum()

        # Compute gradient
        grad = torch.autograd.grad(energy, y, create_graph=True)[0]

        # Compute Hessian
        hessian = []
        for i in range(grad.shape[1]):
            grad_i = torch.autograd.grad(grad[0, i], y, retain_graph=True)[0]
            hessian.append(grad_i.detach().cpu().numpy())

        hessian = np.stack(hessian, axis=0)  # [dim, dim]

        # Eigendecomposition
        if compute_eigenvectors:
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        else:
            eigenvalues = np.linalg.eigvalsh(hessian)
            eigenvectors = None

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        if eigenvectors is not None:
            eigenvectors = eigenvectors[:, idx]

        # Take top k
        eigenvalues = eigenvalues[:self.n_eigenvalues]
        if eigenvectors is not None:
            eigenvectors = eigenvectors[:, :self.n_eigenvalues]

        # Compute metrics
        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 1e-10 else np.inf
        effective_rank = self._compute_effective_rank(eigenvalues)
        spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0.0

        result = HessianAnalysisResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            condition_number=condition_number,
            effective_rank=effective_rank,
            spectral_gap=spectral_gap,
            annealing_level=int(t.item()),
            problem_id=""
        )

        return result

    def _compute_effective_rank(self, eigenvalues: np.ndarray) -> float:
        """
        Compute effective rank using participation ratio.

        Effective rank = (sum λ_i)² / sum λ_i²
        """
        if len(eigenvalues) == 0:
            return 0.0

        eigenvalues = np.abs(eigenvalues)
        sum_eigs = np.sum(eigenvalues)
        sum_sq_eigs = np.sum(eigenvalues ** 2)

        if sum_sq_eigs < 1e-10:
            return 0.0

        return (sum_eigs ** 2) / sum_sq_eigs

    def analyze_eigenspectrum_across_annealing(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        annealing_levels: List[int] = [1, 2, 3, 5, 7, 10]
    ) -> Dict[int, HessianAnalysisResult]:
        """
        Compute Hessian eigenspectrum across multiple annealing levels.

        Args:
            x: Input tensor [batch, inp_dim]
            y: Solution tensor [batch, out_dim]
            annealing_levels: List of k values to analyze

        Returns:
            Dictionary mapping k -> HessianAnalysisResult
        """
        results = {}

        for k in annealing_levels:
            t = torch.tensor([float(k)], device=self.device)
            y_grad = y.clone().detach().requires_grad_(True)

            result = self.compute_hessian_eigenspectrum(x, y_grad, t)
            result.annealing_level = k
            results[k] = result

        return results

    def plot_eigenspectrum(
        self,
        results: Dict[int, HessianAnalysisResult],
        save_path: Optional[str] = None
    ):
        """
        Plot eigenvalue spectrum across annealing levels.

        Args:
            results: Dictionary mapping annealing level -> HessianAnalysisResult
            save_path: Path to save figure (if None, displays)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Eigenvalue magnitude vs index
        ax = axes[0]
        for k, result in sorted(results.items()):
            ax.semilogy(result.eigenvalues, label=f"k={k}", marker='o', markersize=4)
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue (log scale)")
        ax.set_title("Eigenvalue Spectrum Across Annealing")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Condition number vs annealing level
        ax = axes[1]
        levels = sorted(results.keys())
        cond_numbers = [results[k].condition_number for k in levels]
        ax.semilogy(levels, cond_numbers, marker='o')
        ax.set_xlabel("Annealing Level k")
        ax.set_ylabel("Condition Number (log scale)")
        ax.set_title("Condition Number vs Annealing")
        ax.grid(True, alpha=0.3)

        # Plot 3: Effective rank vs annealing level
        ax = axes[2]
        eff_ranks = [results[k].effective_rank for k in levels]
        ax.plot(levels, eff_ranks, marker='o')
        ax.set_xlabel("Annealing Level k")
        ax.set_ylabel("Effective Rank")
        ax.set_title("Effective Rank vs Annealing")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def analyze_eigenvector_alignment(
        self,
        eigenvectors: np.ndarray,
        geometric_basis: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Measure alignment between Hessian eigenvectors and geometric basis.

        Args:
            eigenvectors: Hessian eigenvectors [dim, n_eigs]
            geometric_basis: Geometric basis vectors [dim, n_basis]
                           (e.g., singular vectors, null space basis)
            top_k: Number of top eigenvectors to analyze

        Returns:
            Dictionary with alignment metrics
        """
        # Compute pairwise cosine similarities
        eigenvectors = eigenvectors[:, :top_k]

        # Normalize
        eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=0, keepdims=True) + 1e-10)
        geometric_basis = geometric_basis / (np.linalg.norm(geometric_basis, axis=0, keepdims=True) + 1e-10)

        # Compute alignment matrix
        alignment = np.abs(eigenvectors.T @ geometric_basis)  # [top_k, n_basis]

        # Metrics
        max_alignment = np.max(alignment, axis=1)  # Best alignment for each eigenvector
        mean_alignment = np.mean(max_alignment)

        return {
            "mean_alignment": mean_alignment,
            "max_alignment": np.max(max_alignment),
            "min_alignment": np.min(max_alignment),
            "alignment_matrix": alignment
        }


if __name__ == "__main__":
    # Example usage
    print("Hessian Analysis Module")
    print("Example: Analyze eigenspectrum for matrix completion task")

    # TODO: Load pretrained IRED model and run analysis
