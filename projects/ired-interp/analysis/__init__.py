"""
IRED Energy Field Interpretability Analysis

This package provides tools for analyzing the energy landscapes learned by IRED:
- Hessian eigenspectrum analysis
- Grassmannian manifold geometry
- Sparse autoencoders for gradient features
- Energy landscape visualization
"""

from .hessian_analysis import HessianAnalyzer, HessianAnalysisResult
from .grassmann_geometry import GrassmannianAnalyzer, GeometricAnalysisResult
from .sparse_autoencoder import (
    SparseAutoencoder,
    SAETrainer,
    SAEConfig,
    FeatureAnalyzer,
    GradientDataset,
    collect_gradient_samples
)

__all__ = [
    # Hessian analysis
    "HessianAnalyzer",
    "HessianAnalysisResult",

    # Grassmannian geometry
    "GrassmannianAnalyzer",
    "GeometricAnalysisResult",

    # Sparse autoencoder
    "SparseAutoencoder",
    "SAETrainer",
    "SAEConfig",
    "FeatureAnalyzer",
    "GradientDataset",
    "collect_gradient_samples",
]

__version__ = "0.1.0"
