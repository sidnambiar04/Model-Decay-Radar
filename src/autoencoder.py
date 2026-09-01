import numpy as np
import torch
import torch.nn as nn

# VAE reconstruction error: strictly 0.7 * MSE + 0.3 * KL divergence
MSE_WEIGHT: float = 0.7
KL_WEIGHT: float = 0.3
DEFAULT_LAM: float = MSE_WEIGHT


class AutoencoderDriftDetector(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def combined_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        lam: float = DEFAULT_LAM,
    ) -> torch.Tensor:
        """Per-sample loss: MSE_WEIGHT * MSE + KL_WEIGHT * KL."""
        mse = torch.mean((x - x_hat) ** 2, dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return MSE_WEIGHT * mse + KL_WEIGHT * kl


def train_autoencoder(
    model: AutoencoderDriftDetector,
    reference_data: np.ndarray,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    lam: float = DEFAULT_LAM,
    verbose: bool = True,
) -> AutoencoderDriftDetector:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(reference_data, dtype=torch.float32)
    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        loss_sum = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch = X[idx]
            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch)
            loss = AutoencoderDriftDetector.combined_loss(batch, x_hat, mu, logvar, lam).mean()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch.shape[0]
        if verbose and (epoch == 0 or (epoch+1) % 10 == 0 or epoch == epochs-1):
            print(f"  [AE] epoch {epoch+1}/{epochs} loss: {loss_sum/n:.6f}", flush=True)
    return model


@torch.no_grad()
def compute_reconstruction_errors(
    model: AutoencoderDriftDetector,
    data: np.ndarray,
    lam: float = DEFAULT_LAM,
) -> np.ndarray:
    model.eval()
    X = torch.tensor(data, dtype=torch.float32)
    x_hat, mu, logvar = model(X)
    return AutoencoderDriftDetector.combined_loss(X, x_hat, mu, logvar, lam).numpy()


def compute_dynamic_threshold(ref_errors):
    return float(np.mean(ref_errors) + np.std(ref_errors))
