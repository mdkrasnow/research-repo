"""
Sparse Autoencoder for IRED Energy Gradient Analysis

This module trains sparse autoencoders on gradient field ∇_y E^k(x,y)
to discover interpretable, monosemantic features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder"""
    input_dim: int
    hidden_dim: int
    sparsity_coef: float = 0.01  # L1 regularization coefficient
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 100
    device: str = "cuda"


class GradientDataset(Dataset):
    """Dataset of energy gradients for SAE training"""

    def __init__(self, gradients: torch.Tensor):
        """
        Args:
            gradients: Tensor of shape [N, dim] where N is number of samples
        """
        self.gradients = gradients

    def __len__(self):
        return len(self.gradients)

    def __getitem__(self, idx):
        return self.gradients[idx]


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning interpretable gradient features.

    Architecture:
        Encoder: gradient -> ReLU -> sparse features
        Decoder: sparse features -> linear -> reconstructed gradient

    Loss = reconstruction_loss + sparsity_coef * L1(features)
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # Encoder: gradient -> hidden
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)

        # Decoder: hidden -> gradient
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=True)

        # Initialize decoder as transpose of encoder for tied weights
        # self.decoder.weight.data = self.encoder.weight.data.t()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse features.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Sparse features [batch, hidden_dim]
        """
        features = self.encoder(x)
        features = torch.relu(features)  # ReLU enforces non-negativity
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to reconstructed input.

        Args:
            features: Feature tensor [batch, hidden_dim]

        Returns:
            Reconstructed input [batch, input_dim]
        """
        reconstruction = self.decoder(features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAE.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            (reconstructed_x, features)
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features


class SAETrainer:
    """Trainer for Sparse Autoencoder"""

    def __init__(self, config: SAEConfig):
        self.config = config
        self.model = SparseAutoencoder(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss = reconstruction_loss + sparsity_loss.

        Args:
            x: Original input
            reconstructed: Reconstructed input
            features: Encoded features

        Returns:
            (total_loss, loss_dict)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = nn.functional.mse_loss(reconstructed, x)

        # Sparsity loss (L1 on features)
        sparsity_loss = torch.mean(torch.abs(features))

        # Total loss
        total_loss = reconstruction_loss + self.config.sparsity_coef * sparsity_loss

        loss_dict = {
            "total": total_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "sparsity": sparsity_loss.item(),
            "mean_activation": torch.mean(features).item(),
            "fraction_active": (features > 0).float().mean().item()
        }

        return total_loss, loss_dict

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "sparsity": 0.0,
            "mean_activation": 0.0,
            "fraction_active": 0.0
        }

        for batch in dataloader:
            batch = batch.to(self.config.device)

            # Forward pass
            reconstructed, features = self.model(batch)

            # Compute loss
            loss, loss_dict = self.compute_loss(batch, reconstructed, features)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value

        # Average losses
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        eval_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "sparsity": 0.0,
            "mean_activation": 0.0,
            "fraction_active": 0.0
        }

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.config.device)

                # Forward pass
                reconstructed, features = self.model(batch)

                # Compute loss
                _, loss_dict = self.compute_loss(batch, reconstructed, features)

                # Accumulate losses
                for key, value in loss_dict.items():
                    eval_losses[key] += value

        # Average losses
        num_batches = len(dataloader)
        for key in eval_losses:
            eval_losses[key] /= num_batches

        return eval_losses

    def train(
        self,
        train_dataset: GradientDataset,
        val_dataset: Optional[GradientDataset] = None
    ) -> Dict[str, List[float]]:
        """
        Train SAE.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Returns:
            Dictionary of training history
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )

        history = {
            "train_loss": [],
            "train_reconstruction": [],
            "train_sparsity": [],
            "val_loss": [],
            "val_reconstruction": [],
            "val_sparsity": []
        }

        for epoch in tqdm(range(self.config.num_epochs), desc="Training SAE"):
            # Train epoch
            train_losses = self.train_epoch(train_loader)

            history["train_loss"].append(train_losses["total"])
            history["train_reconstruction"].append(train_losses["reconstruction"])
            history["train_sparsity"].append(train_losses["sparsity"])

            # Validation
            if val_dataset is not None:
                val_losses = self.evaluate(val_loader)
                history["val_loss"].append(val_losses["total"])
                history["val_reconstruction"].append(val_losses["reconstruction"])
                history["val_sparsity"].append(val_losses["sparsity"])

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_losses['total']:.4f} "
                      f"(Recon: {train_losses['reconstruction']:.4f}, "
                      f"Sparsity: {train_losses['sparsity']:.4f})")
                print(f"  Fraction Active: {train_losses['fraction_active']:.3f}")

                if val_dataset is not None:
                    print(f"  Val Loss: {val_losses['total']:.4f}")

        return history

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Total loss
        ax = axes[0]
        ax.plot(history["train_loss"], label="Train")
        if "val_loss" in history and len(history["val_loss"]) > 0:
            ax.plot(history["val_loss"], label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("SAE Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Reconstruction vs Sparsity
        ax = axes[1]
        ax.plot(history["train_reconstruction"], label="Reconstruction")
        ax.plot(history["train_sparsity"], label="Sparsity (scaled)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Component")
        ax.set_title("Loss Components")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class FeatureAnalyzer:
    """Analyze learned SAE features"""

    def __init__(self, sae_model: SparseAutoencoder):
        self.model = sae_model
        self.model.eval()

    def get_feature_activations(
        self,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Get feature activations for input gradients.

        Args:
            gradients: Input gradients [N, input_dim]

        Returns:
            Feature activations [N, hidden_dim]
        """
        with torch.no_grad():
            features = self.model.encode(gradients)
        return features

    def find_top_activating_examples(
        self,
        gradients: torch.Tensor,
        feature_idx: int,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find examples that most activate a specific feature.

        Args:
            gradients: Input gradients [N, input_dim]
            feature_idx: Index of feature to analyze
            top_k: Number of top examples

        Returns:
            (indices, activations) of top-k examples
        """
        features = self.get_feature_activations(gradients)
        activations = features[:, feature_idx]

        # Get top-k
        top_values, top_indices = torch.topk(activations, k=top_k)

        return top_indices, top_values

    def visualize_feature_decoder(
        self,
        feature_idx: int,
        reshape_dims: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize decoder weights for a feature (what the feature encodes).

        Args:
            feature_idx: Feature index
            reshape_dims: If provided, reshape vector to 2D for visualization
            save_path: Path to save figure
        """
        # Get decoder weights for this feature
        decoder_weights = self.model.decoder.weight[:, feature_idx].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        if reshape_dims is not None:
            # Reshape to 2D and plot as heatmap
            decoder_2d = decoder_weights.reshape(reshape_dims)
            im = ax.imshow(decoder_2d, cmap='RdBu_r', aspect='auto')
            ax.set_title(f"Feature {feature_idx} Decoder Weights (2D)")
            plt.colorbar(im, ax=ax)
        else:
            # Plot as 1D
            ax.plot(decoder_weights)
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Weight")
            ax.set_title(f"Feature {feature_idx} Decoder Weights")
            ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def compute_feature_sparsity(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute sparsity of each feature (fraction of examples where active).

        Args:
            features: Feature activations [N, hidden_dim]

        Returns:
            Sparsity per feature [hidden_dim]
        """
        fraction_active = (features > 0).float().mean(dim=0).cpu().numpy()
        return fraction_active


def collect_gradient_samples(
    model: nn.Module,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    t_samples: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Collect energy gradient samples for SAE training.

    Args:
        model: IRED energy model
        x_samples: Input conditions [N, inp_dim]
        y_samples: Candidate solutions [N, out_dim]
        t_samples: Annealing levels [N]
        device: Device

    Returns:
        Gradients [N, out_dim]
    """
    model.eval()
    gradients = []

    for x, y, t in zip(x_samples, y_samples, t_samples):
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device).requires_grad_(True)
        t = t.unsqueeze(0).to(device)

        # Compute energy
        inp = torch.cat([x, y], dim=-1)
        energy = model(inp, t)

        # Compute gradient
        grad = torch.autograd.grad(energy.sum(), y)[0]

        gradients.append(grad.detach().cpu())

    gradients = torch.cat(gradients, dim=0)
    return gradients


if __name__ == "__main__":
    print("Sparse Autoencoder Module")
    print("Example: Train SAE on synthetic gradient data")

    # Generate synthetic data
    torch.manual_seed(42)
    N = 10000
    input_dim = 400
    hidden_dim = 512

    # Create synthetic gradients (should have some structure)
    gradients = torch.randn(N, input_dim) * 0.1

    # Add structured components (simulate interpretable features)
    for i in range(10):
        # Feature i affects specific dimensions
        feature_vector = torch.zeros(input_dim)
        feature_vector[i*40:(i+1)*40] = torch.randn(40)

        # Random activation
        activations = torch.rand(N) > 0.95  # 5% activation (sparse)
        gradients += activations.float().unsqueeze(1) * feature_vector.unsqueeze(0)

    # Create dataset
    dataset = GradientDataset(gradients)

    # Configure and train SAE
    config = SAEConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_coef=0.01,
        num_epochs=50,
        device="cpu"
    )

    trainer = SAETrainer(config)
    history = trainer.train(dataset)

    trainer.plot_training_curves(history)

    print("SAE training complete!")
