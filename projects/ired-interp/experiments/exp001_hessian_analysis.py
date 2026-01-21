"""
Experiment 001: Hessian Eigenspectrum Analysis

Compute Hessian eigenspectrum for matrix completion task across different ranks
and annealing levels.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

from models import EBM, DiffusionWrapper
from dataset import Inverse, Addition
from analysis.hessian_analysis import HessianAnalyzer


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load pretrained IRED model"""
    # Model config for matrix tasks (20x20 = 400 dim)
    inp_dim = 400
    out_dim = 400

    model = EBM(inp_dim=inp_dim, out_dim=out_dim, is_ebm=True)
    wrapper = DiffusionWrapper(model)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        wrapper.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")

    wrapper = wrapper.to(device)
    wrapper.eval()

    return wrapper


def generate_test_problems(
    task: str = "inverse",
    rank: int = 2,
    num_samples: int = 10,
    matrix_size: int = 20
):
    """
    Generate test matrix problems.

    Args:
        task: "inverse", "add", or "completion"
        rank: For completion task
        num_samples: Number of problems to generate
        matrix_size: Size of matrices (20 for 20x20)

    Returns:
        (x_samples, y_samples) tensors
    """
    if task == "inverse":
        dataset = Inverse(split="test", rank=rank, ood=False)
    elif task == "add":
        dataset = Addition(split="test", rank=rank, ood=False)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Sample problems
    x_samples = []
    y_samples = []

    for i in range(min(num_samples, len(dataset))):
        x, y = dataset[i]
        x_samples.append(torch.from_numpy(x).float())
        y_samples.append(torch.from_numpy(y).float())

    x_samples = torch.stack(x_samples)
    y_samples = torch.stack(y_samples)

    return x_samples, y_samples


def run_hessian_analysis(
    model: torch.nn.Module,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    annealing_levels: list = [1, 3, 5, 7, 10],
    device: str = "cuda",
    output_dir: str = "results/exp001"
):
    """
    Run Hessian analysis on samples.

    Args:
        model: IRED energy model
        x_samples: Input conditions [N, inp_dim]
        y_samples: Solutions [N, out_dim]
        annealing_levels: List of k values to analyze
        device: Device
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = HessianAnalyzer(
        model=model.ebm,  # Extract EBM from wrapper
        device=device,
        n_eigenvalues=20,
        use_lanczos=True
    )

    # Results storage
    all_results = {}

    # Analyze each sample
    for sample_idx in range(len(x_samples)):
        print(f"\n=== Analyzing sample {sample_idx+1}/{len(x_samples)} ===")

        x = x_samples[sample_idx:sample_idx+1]
        y = y_samples[sample_idx:sample_idx+1]

        # Analyze across annealing levels
        results = analyzer.analyze_eigenspectrum_across_annealing(
            x, y, annealing_levels
        )

        # Save results for this sample
        sample_results = {}
        for k, result in results.items():
            sample_results[k] = {
                "eigenvalues": result.eigenvalues.tolist(),
                "condition_number": result.condition_number,
                "effective_rank": result.effective_rank,
                "spectral_gap": result.spectral_gap
            }

        all_results[f"sample_{sample_idx}"] = sample_results

        # Plot for this sample
        save_path = os.path.join(output_dir, f"hessian_sample_{sample_idx}.png")
        analyzer.plot_eigenspectrum(results, save_path=save_path)
        print(f"Saved plot to {save_path}")

    # Save all results
    results_path = os.path.join(output_dir, "hessian_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {results_path}")

    # Compute aggregate statistics
    compute_aggregate_stats(all_results, output_dir)


def compute_aggregate_stats(all_results: dict, output_dir: str):
    """Compute aggregate statistics across samples"""
    import matplotlib.pyplot as plt

    # Extract data
    annealing_levels = sorted([int(k) for k in list(all_results.values())[0].keys()])

    # Aggregate metrics
    avg_condition_numbers = {k: [] for k in annealing_levels}
    avg_effective_ranks = {k: [] for k in annealing_levels}

    for sample_id, sample_results in all_results.items():
        for k in annealing_levels:
            k_str = str(k)
            avg_condition_numbers[k].append(sample_results[k_str]["condition_number"])
            avg_effective_ranks[k].append(sample_results[k_str]["effective_rank"])

    # Compute means and stds
    condition_means = [np.mean(avg_condition_numbers[k]) for k in annealing_levels]
    condition_stds = [np.std(avg_condition_numbers[k]) for k in annealing_levels]

    rank_means = [np.mean(avg_effective_ranks[k]) for k in annealing_levels]
    rank_stds = [np.std(avg_effective_ranks[k]) for k in annealing_levels]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(annealing_levels, condition_means, yerr=condition_stds,
                marker='o', capsize=5)
    ax.set_xlabel("Annealing Level k")
    ax.set_ylabel("Condition Number")
    ax.set_title("Average Condition Number vs Annealing")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(annealing_levels, rank_means, yerr=rank_stds,
                marker='o', capsize=5)
    ax.set_xlabel("Annealing Level k")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Average Effective Rank vs Annealing")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "aggregate_stats.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregate statistics to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="EXP-001: Hessian Eigenspectrum Analysis")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/matrix_inverse.pt",
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--task", type=str, default="inverse",
                       choices=["inverse", "add"],
                       help="Matrix task to analyze")
    parser.add_argument("--rank", type=int, default=2,
                       help="Matrix rank for completion task")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of test samples to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="results/exp001",
                       help="Output directory")

    args = parser.parse_args()

    print("="*60)
    print("Experiment 001: Hessian Eigenspectrum Analysis")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Rank: {args.rank}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print("="*60)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device=args.device)

    # Generate test problems
    print("\nGenerating test problems...")
    x_samples, y_samples = generate_test_problems(
        task=args.task,
        rank=args.rank,
        num_samples=args.num_samples
    )

    x_samples = x_samples.to(args.device)
    y_samples = y_samples.to(args.device)

    print(f"Generated {len(x_samples)} test problems")
    print(f"Input shape: {x_samples.shape}")
    print(f"Output shape: {y_samples.shape}")

    # Run Hessian analysis
    print("\nRunning Hessian analysis...")
    annealing_levels = [1, 3, 5, 7, 10]

    run_hessian_analysis(
        model=model,
        x_samples=x_samples,
        y_samples=y_samples,
        annealing_levels=annealing_levels,
        device=args.device,
        output_dir=output_dir
    )

    print("\n" + "="*60)
    print("Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
