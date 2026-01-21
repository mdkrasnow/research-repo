#!/usr/bin/env python3
"""
IRED Baseline Experiment Runner

Runs the official IRED implementation on matrix inversion task.

Usage:
    python experiments/run_baseline.py --config configs/q001_pilot.json --output-dir runs/q001_run
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from dataset import Inverse
from models import DiffusionWrapper, EBM
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D


def load_config(config_path):
    """Load experiment configuration from JSON."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def create_model(config):
    """Create energy-based model from config."""
    model_cfg = config.get('model', {})

    # Create base EBM
    input_dim = model_cfg.get('input_dim', 100)
    output_dim = model_cfg.get('output_dim', 100)
    hidden_dim = model_cfg.get('hidden_dim', 512)

    ebm = EBM(input_dim, output_dim, hidden_dim)

    # Wrap in diffusion wrapper
    model = DiffusionWrapper(ebm)

    return model


def create_dataset(config):
    """Create dataset from config."""
    dataset_cfg = config.get('dataset', {})
    task = config.get('task', 'inverse')
    seed = config.get('seed', 42)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if task == 'inverse':
        rank = dataset_cfg.get('rank', 10)
        ood = dataset_cfg.get('ood', False)
        split = 'train'  # Default to train split
        dataset = Inverse(split=split, rank=rank, ood=ood)
    else:
        raise ValueError(f"Unknown task: {task}")

    return dataset


def create_diffusion(model, config):
    """Create diffusion model from config."""
    diffusion_cfg = config.get('diffusion', {})

    steps = diffusion_cfg.get('steps', 10)
    beta_schedule = diffusion_cfg.get('beta_schedule', 'cosine')

    # Create diffusion model
    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=config['dataset']['rank'] ** 2,  # Flattened matrix
        timesteps=steps,
        beta_schedule=beta_schedule
    )

    return diffusion


def run_experiment(config, output_dir):
    """Run IRED baseline experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting IRED baseline experiment")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Output directory: {output_dir}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model, dataset, diffusion
    print("\n[1/4] Creating model...")
    model = create_model(config)
    model = model.to(device)

    print("[2/4] Creating dataset...")
    dataset = create_dataset(config)

    print("[3/4] Creating diffusion model...")
    diffusion = create_diffusion(model, config)
    diffusion = diffusion.to(device)

    print("[4/4] Setting up trainer...")
    training_cfg = config.get('training', {})
    batch_size = training_cfg.get('batch_size', 256)
    lr = training_cfg.get('learning_rate', 1e-4)
    train_steps = training_cfg.get('train_steps', 100000)
    ema_decay = training_cfg.get('ema_decay', 0.995)

    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_lr=lr,
        train_num_steps=train_steps,
        ema_decay=ema_decay,
        amp=False  # Disable mixed precision for stability
    )

    # Train
    print(f"\nTraining for {train_steps} steps...")
    trainer.train()

    # Save results
    results = {
        "status": "completed",
        "config": config,
        "device": str(device),
        "notes": "Official IRED baseline experiment completed successfully"
    }

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Experiment completed successfully")
    print(f"✓ Results saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to config JSON')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    args = parser.parse_args()

    # Validate paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load and run
    config = load_config(config_path)
    results = run_experiment(config, args.output_dir)

    print(f"\nFinal status: {results['status']}")


if __name__ == '__main__':
    main()
