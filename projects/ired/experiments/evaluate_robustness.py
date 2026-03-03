#!/usr/bin/env python3
"""
Robustness evaluation script for IRED matrix inversion models.

Evaluates trained models on test sets using the same diffusion sampling
procedure used during training validation. This ensures MSE values are
directly comparable to training-time validation metrics.

Usage:
    python experiments/evaluate_robustness.py --config configs/q225_ood_evaluation.json
    python experiments/evaluate_robustness.py --config configs/q225_ood_evaluation.json --seed 0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add ired directory to path
ired_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ired_dir))

import torch
import numpy as np
from dataset import Inverse
from models import EBM, DiffusionWrapper
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D


def load_config(config_path):
    """Load evaluation configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def load_diffusion_model(checkpoint_path, config, device='cpu'):
    """
    Load a trained model into a full GaussianDiffusion1D for proper
    diffusion-based sampling (matching training-time validation).
    """
    print(f"  Loading checkpoint from: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location=device)

    # Build EBM + DiffusionWrapper (same as training)
    inp_dim = 400
    out_dim = 400
    use_scalar_energy = config.get('use_scalar_energy', True)
    ebm = EBM(inp_dim=inp_dim, out_dim=out_dim, is_ebm=True, use_scalar_energy=use_scalar_energy)
    grad_norm_ref = config.get('batch_size', None)
    wrapper = DiffusionWrapper(ebm, grad_norm_ref=grad_norm_ref)

    # Build GaussianDiffusion1D (same params as training)
    diffusion_steps = config.get('diffusion_steps', 10)
    inference_opt_steps = config.get('inference_opt_steps', None)
    diffusion = GaussianDiffusion1D(
        wrapper,
        seq_length=32,
        objective='pred_noise',
        timesteps=diffusion_steps,
        continuous=True,
        inference_opt_steps=inference_opt_steps,
    )

    # Load weights — prefer EMA from Trainer1D milestone checkpoints (model-N.pt)
    if 'ema' in checkpoint_data:
        # Trainer1D milestone checkpoint: extract EMA model weights
        ema_state = checkpoint_data['ema']
        # EMA keys are 'ema_model.X' — strip prefix to get GaussianDiffusion1D state
        diffusion_state = {}
        for key, val in ema_state.items():
            if key.startswith('ema_model.'):
                diffusion_state[key[len('ema_model.'):]] = val
        diffusion.load_state_dict(diffusion_state, strict=False)
        print(f"  ✓ Loaded {len(diffusion_state)} EMA parameters from milestone checkpoint")
    elif 'model' in checkpoint_data:
        # Trainer1D milestone without EMA — use raw model weights
        diffusion.load_state_dict(checkpoint_data['model'], strict=False)
        print(f"  ✓ Loaded model parameters from checkpoint")
    else:
        raise RuntimeError(f"Unrecognized checkpoint format. Keys: {list(checkpoint_data.keys())}")

    diffusion = diffusion.to(device)
    diffusion.eval()
    opt_steps_str = inference_opt_steps if inference_opt_steps is not None else "5 (default)"
    print(f"  ✓ GaussianDiffusion1D built (timesteps={diffusion_steps}, inference_opt_steps={opt_steps_str}, continuous=True)")

    return diffusion


def evaluate_with_diffusion_sampling(diffusion, test_loader, device='cpu'):
    """
    Evaluate model using the same diffusion sampling as training validation.
    """
    mse_list = []

    print(f"  Running evaluation with diffusion sampling...")
    with torch.no_grad():
        for batch_idx, (matrices, true_inverses) in enumerate(test_loader):
            matrices = matrices.to(device).float()
            true_inverses = true_inverses.to(device).float()

            # Use diffusion sample() — same as Trainer1D._run_validation
            samples = diffusion.sample(
                matrices, true_inverses, None,
                batch_size=matrices.size(0)
            )

            # Compute per-sample MSE
            mse = (samples - true_inverses).pow(2).mean(dim=1)
            mse_list.extend(mse.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                current_mse = np.mean(mse_list)
                print(f"    Batch {batch_idx + 1}: current MSE = {current_mse:.6f}")

    mse_array = np.array(mse_list)

    result = {
        "mse_mean": float(np.mean(mse_array)),
        "mse_std": float(np.std(mse_array)),
        "mse_median": float(np.median(mse_array)),
        "mse_min": float(np.min(mse_array)),
        "mse_max": float(np.max(mse_array)),
        "percent_solved": float(100.0 * (mse_array < 0.01).mean()),
        "num_samples": int(len(mse_array)),
    }

    return result


def evaluate_robustness(config):
    """Run robustness evaluation for specified difficulty level."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {config['experiment_name']}")
    print(f"{'='*80}")
    print(f"Difficulty: ood={config['ood']}, rank={config['rank']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load test dataset
    print(f"Loading test dataset...")
    test_dataset = Inverse(
        split='test',
        rank=config['rank'],
        ood=config['ood']
    )
    print(f"  Input dimension: {test_dataset.inp_dim}, Output dimension: {test_dataset.out_dim}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=False,
        num_workers=0
    )

    # Evaluate each checkpoint
    results = {}
    for checkpoint_path in config['model_checkpoints']:
        model_name = Path(checkpoint_path).parent.name
        print(f"\nEvaluating model: {model_name}")

        try:
            diffusion = load_diffusion_model(checkpoint_path, config, device=device)

            model_results = evaluate_with_diffusion_sampling(
                diffusion, test_loader, device=device
            )

            results[model_name] = model_results
            print(f"  ✓ MSE: {model_results['mse_mean']:.6f} ± {model_results['mse_std']:.6f}")
            print(f"  ✓ % Solved (MSE<0.01): {model_results['percent_solved']:.1f}%")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Robustness evaluation for IRED models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON evaluation config file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--inference-opt-steps', type=int, default=None,
                        help='Override inner-loop optimization steps per landscape at inference')

    args = parser.parse_args()

    config = load_config(args.config)

    if args.inference_opt_steps is not None:
        config['inference_opt_steps'] = args.inference_opt_steps
        print(f"CLI override: inference_opt_steps={args.inference_opt_steps}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    if args.output_dir:
        config['output_dir'] = args.output_dir

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(json.dumps(config, indent=2))

    results = evaluate_robustness(config)

    # Save results
    results_file = output_dir / f"{config['experiment_name']}_results.json"
    print(f"\nSaving results to: {results_file}")
    with open(results_file, 'w') as f:
        json.dump({
            "experiment": config['experiment_name'],
            "difficulty_level": f"ood={config['ood']}",
            "rank": config['rank'],
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Summary: {config['experiment_name']}")
    print(f"{'='*80}")
    for model_name, model_results in results.items():
        if "error" not in model_results:
            print(f"\n{model_name}:")
            print(f"  MSE: {model_results['mse_mean']:.6f} ± {model_results['mse_std']:.6f}")
            print(f"  % Solved: {model_results['percent_solved']:.1f}%")
        else:
            print(f"\n{model_name}: ERROR - {model_results['error']}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
