#!/usr/bin/env python3
"""
Robustness evaluation script for IRED matrix inversion models.

Evaluates trained models on difficulty-scaled test sets to measure robustness
degradation when moving from well-conditioned (ood=false) to ill-conditioned
(ood=true) matrices.

Usage:
    python experiments/evaluate_robustness.py --config configs/q224_eval_difficulty_L1_baseline.json
    python experiments/evaluate_robustness.py --config configs/q224_eval_difficulty_L2_hard.json
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


def load_config(config_path):
    """Load evaluation configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def load_model_from_checkpoint(checkpoint_path, inp_dim, out_dim, grad_norm_ref=None, device='cpu'):
    """Load a trained EBM model from a trainer checkpoint."""
    print(f"  Loading model from: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load the checkpoint (saved by Trainer1D or matrix_inversion_mining.py)
    checkpoint_data = torch.load(checkpoint_path, map_location=device)

    # Initialize EBM and wrapper
    ebm = EBM(inp_dim=inp_dim, out_dim=out_dim, is_ebm=True, use_scalar_energy=True)
    model = DiffusionWrapper(ebm, grad_norm_ref=grad_norm_ref)

    # Handle new checkpoint format (clean format with 'ebm_state' key)
    if 'ebm_state' in checkpoint_data:
        ebm_state = checkpoint_data['ebm_state']
        print(f"  Loading new checkpoint format (clean EBM state)")
    else:
        # Fall back to old format: extract from full state dict
        if 'model' in checkpoint_data:
            state_dict = checkpoint_data['model']
        else:
            state_dict = checkpoint_data

        # Filter to only EBM-related keys
        ebm_state = {}
        for key, val in state_dict.items():
            # Skip diffusion parameters
            if any(skip in key for skip in ['betas', 'alphas_cumprod', 'posterior_',
                                             'opt_step_size', 'loss_weight',
                                             'sqrt_alphas', 'sqrt_one_minus', 'log_one_minus']):
                continue
            # Extract EBM weights from 'model.ebm.*' or 'ebm.*' keys
            if key.startswith('model.ebm.'):
                ebm_key = key[6:]  # Remove 'model.' prefix
                ebm_state[ebm_key] = val
            elif key.startswith('ebm.'):
                ebm_state[key] = val

    # Load EBM parameters
    if ebm_state:
        try:
            model.ebm.load_state_dict(ebm_state, strict=False)
            print(f"  ✓ Loaded {len(ebm_state)} EBM parameters")
        except Exception as e:
            print(f"  Warning: Error loading EBM state: {e}")
    else:
        print(f"  Warning: No EBM parameters found in checkpoint")

    model = model.to(device)
    model.eval()

    return model


def inference_with_optimization(model, inp, num_opt_steps=10, device='cpu'):
    """
    Run optimization on the input using the trained EBM model.

    The EBM model provides gradients that minimize energy. We use gradient descent
    to iteratively optimize the output toward the solution.

    Args:
        model: DiffusionWrapper with trained EBM
        inp: Input matrices of shape [B, inp_dim]
        num_opt_steps: Number of optimization iterations
        device: torch device

    Returns:
        Predicted outputs of shape [B, out_dim]
    """
    B, _ = inp.shape
    out_dim = model.out_dim

    # Initialize with random guesses or zeros
    with torch.no_grad():
        predictions = torch.randn(B, out_dim, device=device) * 0.1

    # Optimization loop
    for step in range(num_opt_steps):
        predictions = predictions.detach()
        predictions.requires_grad_(True)

        # Get gradients from the model
        try:
            with torch.enable_grad():
                t = torch.zeros(B, dtype=torch.long, device=device)
                grads = model(inp, predictions, t=t)
        except Exception as e:
            print(f"    Error during forward pass at step {step}: {e}")
            print(f"      Type of model: {type(model)}")
            import traceback
            traceback.print_exc()
            raise

        # Gradient descent step
        step_size = 0.01
        with torch.no_grad():
            predictions = predictions - step_size * grads

    return predictions.detach()


def evaluate_on_difficulty(model, test_loader, num_opt_steps=10, device='cpu'):
    """
    Evaluate model on test set at a given difficulty level.

    Returns:
        dict with MSE statistics and condition number info
    """
    mse_list = []
    condition_numbers = []

    print(f"  Running evaluation...")
    print(f"    num_opt_steps type: {type(num_opt_steps)}, value: {num_opt_steps}")
    print(f"    model type: {type(model)}")
    print(f"    model.ebm type: {type(model.ebm)}")
    print(f"    model.grad_norm_ref: {model.grad_norm_ref}")

    with torch.no_grad():
        for batch_idx, (matrices, true_inverses) in enumerate(test_loader):
            matrices = matrices.to(device).float()
            true_inverses = true_inverses.to(device).float()

            # Compute condition numbers for context (skip for now - causing errors)
            # Will compute at the end on the full tensor batch
            pass

            # Run optimization
            try:
                pred_inverses = inference_with_optimization(
                    model,
                    matrices,
                    num_opt_steps=num_opt_steps,
                    device=device
                )
            except Exception as e:
                print(f"    Error at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Compute MSE
            mse = torch.mean((pred_inverses - true_inverses) ** 2, dim=1)
            mse_list.extend(mse.detach().cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                current_mse = np.mean(mse_list)
                print(f"    Batch {batch_idx + 1}: current MSE = {current_mse:.6f}")

    mse_array = np.array(mse_list)
    cond_array = np.array(condition_numbers) if condition_numbers else np.array([])

    result = {
        "mse_mean": float(np.mean(mse_array)),
        "mse_std": float(np.std(mse_array)),
        "mse_median": float(np.median(mse_array)),
        "mse_min": float(np.min(mse_array)),
        "mse_max": float(np.max(mse_array)),
        "percent_solved": float(100.0 * (mse_array < 0.01).mean()),
        "num_samples": int(len(mse_array)),
    }

    # Add condition number stats if available
    if len(cond_array) > 0:
        result.update({
            "condition_number_mean": float(np.mean(cond_array)),
            "condition_number_std": float(np.std(cond_array)),
            "condition_number_median": float(np.median(cond_array)),
        })
    else:
        result.update({
            "condition_number_mean": None,
            "condition_number_std": None,
            "condition_number_median": None,
        })

    return result


def evaluate_robustness(config):
    """
    Run robustness evaluation for specified difficulty level.

    Args:
        config: Evaluation configuration dict

    Returns:
        dict with results for each model at this difficulty level
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {config['experiment_name']}")
    print(f"{'='*80}")
    print(f"Difficulty: ood={config['ood']}, rank={config['rank']}")

    # Device setup
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

    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=False,
        num_workers=0
    )

    # Evaluate each model
    results = {}
    for checkpoint_path in config['model_checkpoints']:
        # Extract model name from path
        model_name = Path(checkpoint_path).parent.name
        print(f"\nEvaluating model: {model_name}")

        try:
            # Load model
            model = load_model_from_checkpoint(
                checkpoint_path,
                inp_dim=test_dataset.inp_dim,
                out_dim=test_dataset.out_dim,
                grad_norm_ref=None,  # grad_norm_ref should be None or a callable, not batch_size
                device=device
            )

            # Evaluate
            model_results = evaluate_on_difficulty(
                model,
                test_loader,
                num_opt_steps=config.get('num_opt_steps', 10),
                device=device
            )

            results[model_name] = model_results
            print(f"  ✓ MSE: {model_results['mse_mean']:.6f} ± {model_results['mse_std']:.6f}")
            print(f"  ✓ % Solved (MSE<0.01): {model_results['percent_solved']:.1f}%")
            print(f"  ✓ Condition #: {model_results['condition_number_mean']:.2f} ± {model_results['condition_number_std']:.2f}")

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

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(json.dumps(config, indent=2))

    # Run evaluation
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
            print(f"  Condition #: {model_results['condition_number_mean']:.2f}")
        else:
            print(f"\n{model_name}: ERROR - {model_results['error']}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
