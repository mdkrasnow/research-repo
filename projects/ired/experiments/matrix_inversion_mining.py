#!/usr/bin/env python3
"""
Experiment runner for adversarial negative mining on matrix inversion task.

Usage:
    python experiments/matrix_inversion_mining.py --config configs/q001_baseline.json
    python experiments/matrix_inversion_mining.py --mining-strategy adversarial --rank 20 --train-steps 100000

This script reuses the IRED diffusion infrastructure (Trainer1D, GaussianDiffusion1D)
and adds configurable negative mining strategies.
"""

import os
import os.path as osp
import sys
import argparse
import json
from pathlib import Path

# Add ired directory to path so we can import dataset, models, etc.
ired_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ired_dir))

import torch
import numpy as np
import random
from dataset import Inverse
from models import EBM, DiffusionWrapper
from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D


def str2bool(x):
    """Convert string to boolean"""
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError(f'Invalid boolean value: {x}')


def load_config(config_path):
    """Load experiment configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def run_experiment(config):
    """
    Run matrix inversion experiment with specified mining strategy.

    Args:
        config: Dictionary containing experiment configuration
            - mining_strategy: 'none', 'random', or 'adversarial'
            - rank: Matrix dimension (rank x rank)
            - diffusion_steps: Number of diffusion timesteps
            - batch_size: Training batch size
            - train_steps: Total training iterations
            - learning_rate: Optimizer learning rate
            - seed: Random seed for reproducibility (optional)
            - output_dir: Directory for results and checkpoints
    """
    # Set random seed if provided
    seed = config.get('seed', None)
    if seed is not None:
        print(f"Setting random seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print(f"Starting experiment with mining strategy: {config['mining_strategy']}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize dataset
    rank = config.get('rank', 20)
    ood = config.get('ood', False)
    dataset = Inverse(split='train', rank=rank, ood=ood)
    validation_dataset = dataset  # Use same dataset for validation
    
    print(f"Dataset: Inverse matrices ({rank}x{rank})")
    print(f"Input dimension: {dataset.inp_dim}, Output dimension: {dataset.out_dim}")
    
    # Initialize model
    use_scalar_energy = config.get('use_scalar_energy', False)
    model = EBM(inp_dim=dataset.inp_dim, out_dim=dataset.out_dim, is_ebm=True, use_scalar_energy=use_scalar_energy)
    model = DiffusionWrapper(model)
    
    # Initialize diffusion model with mining configuration
    diffusion_steps = config.get('diffusion_steps', 10)
    mining_strategy = config.get('mining_strategy', 'none')
    
    # NOTE: This will require modifications to GaussianDiffusion1D (Task T1)
    # For now, we pass mining_strategy through kwargs
    mining_config = {
        # Core parameters
        'strategy': mining_strategy,
        'opt_steps': config.get('mining_opt_steps', 2),
        'noise_scale': config.get('mining_noise_scale', 3.0),

        # CD core features
        'use_cd_loss': config.get('use_cd_loss', False),
        'use_langevin': config.get('use_langevin', False),
        'langevin_sigma_multiplier': config.get('langevin_sigma_multiplier', 1.0),
        'langevin_grad_clip': config.get('langevin_grad_clip', 0.01),
        'energy_loss_weight': config.get('energy_loss_weight', 0.05),
        'energy_reg_weight': config.get('energy_reg_weight', 0.1),

        # Replay buffer
        'use_replay_buffer': config.get('use_replay_buffer', False),
        'replay_buffer_size': config.get('replay_buffer_size', 10000),
        'replay_buffer_buckets': config.get('replay_buffer_buckets', 16),
        'replay_sample_prob': config.get('replay_sample_prob', 0.95),

        # Residual filtering
        'use_residual_filter': config.get('use_residual_filter', False),
        'residual_filter_quantile': config.get('residual_filter_quantile', 0.3),

        # Energy scheduling
        'use_energy_schedule': config.get('use_energy_schedule', False),
        'energy_loss_warmup_steps': config.get('energy_loss_warmup_steps', 20000),
        'energy_loss_max_weight': config.get('energy_loss_max_weight', 0.05),

        # Timestep range filtering
        'use_timestep_range': config.get('use_timestep_range', False),
        'energy_loss_timestep_range': config.get('energy_loss_timestep_range', [0.2, 0.8]),

        # IRED-style contrastive loss
        'use_ired_contrastive_loss': config.get('use_ired_contrastive_loss', False),
        'contrastive_temperature': config.get('contrastive_temperature', 1.0)
    }
    
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=32,
        objective='pred_noise',
        timesteps=diffusion_steps,
        continuous=True,
        mining_config=mining_config  # Custom parameter (requires T1 implementation)
    )
    
    # Initialize trainer
    batch_size = config.get('batch_size', 2048)
    train_steps = config.get('train_steps', 100000)
    learning_rate = config.get('learning_rate', 1e-4)
    output_dir = config.get('output_dir', 'results/ds_inverse/model_mlp')
    
    # Check if GPU is available for fp16
    use_fp16 = torch.cuda.is_available()

    # Adam beta1=0 for CD-style EBM training.
    # UvA DL Tutorial 8 (Lippe 2022) sets beta1=0 when training EBMs with MCMC
    # because momentum accumulates gradient direction from past steps, but in EBM
    # training the gradient direction changes substantially each step (the negative
    # phase samples a new set of negatives every iteration). Carrying stale momentum
    # from a previous negative set biases the parameter update. beta1=0 makes the
    # first-moment estimate equal to the current gradient, disabling this momentum.
    # For the baseline (no CD loss), beta1=0.9 is standard; the config overrides this
    # only for CD experiments (q202-q204 set "adam_betas": [0.0, 0.999]).
    adam_betas_cfg = config.get('adam_betas', None)
    adam_betas = tuple(adam_betas_cfg) if adam_betas_cfg is not None else (0.9, 0.99)

    trainer = Trainer1D(
        diffusion,
        dataset,
        train_batch_size=batch_size,
        validation_batch_size=256,
        train_lr=learning_rate,
        train_num_steps=train_steps,
        gradient_accumulate_every=1,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=adam_betas,
        save_and_sample_every=1000,
        results_folder=output_dir,
        metric='mse',
        validation_dataset=validation_dataset,
        fp16=use_fp16,  # Mixed precision only on CUDA GPUs
        split_batches=True
    )
    
    print(f"Trainer initialized. Output directory: {output_dir}")
    print(f"Training for {train_steps} steps...")
    
    # Train
    trainer.train()
    
    print("Training completed!")
    
    # Return summary
    results = {
        "status": "completed",
        "mining_strategy": mining_strategy,
        "final_step": trainer.step,
        "output_dir": str(output_dir)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Matrix inversion experiment with adversarial negative mining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Configuration source
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides individual args)')
    
    # Core experiment parameters
    parser.add_argument('--mining-strategy', type=str, default='none',
                        choices=['none', 'random', 'adversarial'],
                        help='Negative mining strategy')
    parser.add_argument('--rank', type=int, default=20,
                        help='Matrix dimension (rank x rank)')
    parser.add_argument('--ood', type=str2bool, default=False,
                        help='Use out-of-distribution test matrices')
    
    # Training hyperparameters
    parser.add_argument('--diffusion-steps', type=int, default=10,
                        help='Number of diffusion timesteps')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Training batch size')
    parser.add_argument('--train-steps', type=int, default=100000,
                        help='Total training iterations')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Optimizer learning rate')
    
    # Mining-specific parameters
    parser.add_argument('--mining-opt-steps', type=int, default=2,
                        help='Number of gradient steps for adversarial mining')
    parser.add_argument('--mining-noise-scale', type=float, default=3.0,
                        help='Noise scale for negative sample initialization')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        # Allow command-line seed to override config file seed
        if args.seed is not None:
            config['seed'] = args.seed
    else:
        # Build config from command-line arguments
        config = {
            'mining_strategy': args.mining_strategy,
            'rank': args.rank,
            'ood': args.ood,
            'diffusion_steps': args.diffusion_steps,
            'batch_size': args.batch_size,
            'train_steps': args.train_steps,
            'learning_rate': args.learning_rate,
            'mining_opt_steps': args.mining_opt_steps,
            'mining_noise_scale': args.mining_noise_scale,
            'seed': args.seed,
        }

    # Set output directory
    if args.output_dir:
        config['output_dir'] = args.output_dir
    elif 'output_dir' not in config:
        # Auto-generate output directory name
        strategy = config['mining_strategy']
        rank = config['rank']
        config['output_dir'] = f'results/ds_inverse/model_mlp_mining_{strategy}_rank_{rank}'
    
    # Ensure output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save config to output directory
    config_save_path = osp.join(config['output_dir'], 'experiment_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_save_path}")
    
    # Run experiment
    results = run_experiment(config)
    
    # Save results
    results_path = osp.join(config['output_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
