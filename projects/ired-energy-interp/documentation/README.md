# IRED Energy Interpretability: Documentation

This directory contains project documentation for the IRED energy dynamics and interpretability research project.

## Files

- **implementation-todo.md**: Detailed task breakdown for all 7 research phases
- **queue.md**: Current experiment queue and submission status
- **debugging.md**: Issues encountered and resolutions
- **experimental-design.md**: Experimental design decisions and methodology
- **decision-summary.md**: Key project decisions and rationale

## Project Overview

**Goal**: Understand how IRED's learned energy function encodes matrix invertibility, and discover interpretable directions in the energy landscape that correspond to real matrix geometric properties.

**Approach**: Apply state-of-art mechanistic interpretability techniques from 2024-2025:
- Sparse Autoencoders (SAEs) for feature decomposition
- Integrated Gradients for attribution analysis
- Concept Activation Vectors (CAVs) for semantic directions
- Riemannian geometry on SPD manifolds
- Causal intervention validation
- Influence functions for robustness

**Phases**:
1. Energy Landscape Characterization
2. Sparse Autoencoder Decomposition
3. Gradient Attribution & Interpretable Directions
4. Riemannian Geometry Analysis
5. Layer-wise Compositional Analysis
6. Robustness Validation & Adversarial Testing
7. Integration with Adversarial Mining

## Quick Links

- Research Question: [`../research_question.md`](../research_question.md)
- Implementation Tasks: [`implementation-todo.md`](implementation-todo.md)
- Pipeline Status: [`.state/pipeline.json`](../.state/pipeline.json)
