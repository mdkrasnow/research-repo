"""
Tiny symmetry learner: encoder -> learned action matrix A(delta) -> decoder.

The action-matrix structure is what lets us TEST whether the model learned a real
group action (composition / equivariance) vs. a shortcut. A pure black-box MLP would
hide the structure; this factorization exposes it.

    z      = Encoder(input)                  in R^latent
    A      = ActionNet([sin d, cos d])       in R^{latent x latent}
    z_next = A @ z
    y_hat  = Decoder(z_next)                  in R^2
"""

import torch
import torch.nn as nn


class SymmetryLearner(nn.Module):
    def __init__(self, input_dim=6, latent_dim=2, hidden=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.action_net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def action_matrix(self, delta):
        d = torch.stack([torch.sin(delta), torch.cos(delta)], dim=-1)   # [n,2]
        A = self.action_net(d)
        return A.view(-1, self.latent_dim, self.latent_dim)

    def encode(self, inp):
        return self.encoder(inp)

    def forward(self, inp, delta):
        z = self.encoder(inp)
        A = self.action_matrix(delta)
        z_next = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)
        return self.decoder(z_next)
