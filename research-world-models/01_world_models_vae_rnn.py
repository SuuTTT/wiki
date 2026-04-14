import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    env_id: str = "CarRacing-v2"
    
    # VAE parameters
    latent_dim: int = 32
    vae_lr: float = 1e-4
    vae_batch_size: int = 100
    
    # MDN-RNN parameters
    rnn_hidden_dim: int = 256
    num_gaussian_mix: int = 5
    rnn_lr: float = 1e-3
    seq_len: int = 1000

class VAE(nn.Module):
    """
    Vision Model (V):
    Compresses high-dimensional visual observations into a low-dimensional latent vector z.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: (Batch, 3, 64, 64) -> (Batch, 256) (assuming resized observations)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        # Mu and logvar for the reparameterization trick
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim) # Example dimensions, requires tuning to env
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        
        # Decoder: (Batch, latent_dim) -> (Batch, 3, 64, 64)
        self.fc_decode = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 2, 2)), # Example unflatten
            nn.ConvTranspose2d(256, 128, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

class MDNRNN(nn.Module):
    """
    Memory Model (M):
    Predicts the next latent state z_{t+1} as a probability distribution given the 
    current latent state z_t, action a_t, and previous hidden state h_t.
    Uses a Mixture Density Network (MDN) on top of an LSTM.
    """
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_mix=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mix = num_mix
        self.latent_dim = latent_dim
        
        self.lstm = nn.LSTM(input_size=latent_dim + action_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Outputs parameters for the Mixture of Gaussians: pi (weights), mu (means), sigma (std devs)
        self.fc_pi = nn.Linear(hidden_dim, num_mix * latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, num_mix * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_mix * latent_dim)

    def forward(self, z, action, hidden):
        # z: (batch, seq_len, latent_dim), action: (batch, seq_len, action_dim)
        x = torch.cat([z, action], dim=-1)
        out, hidden = self.lstm(x, hidden)
        
        # MDN layer
        pi = self.fc_pi(out).view(-1, self.num_mix, self.latent_dim)
        pi = F.softmax(pi, dim=1) # mixture weights
        mu = self.fc_mu(out).view(-1, self.num_mix, self.latent_dim)
        sigma = torch.exp(self.fc_sigma(out)).view(-1, self.num_mix, self.latent_dim)
        
        return pi, mu, sigma, hidden

class Controller(nn.Module):
    """
    Controller Model (C):
    Maps the concatenated latent state `z_t` and RNN hidden state `h_t` to an action `a_t`.
    In the original World Models paper, this is trained via evolutionary strategies (CMA-ES).
    """
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)
        
    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return self.fc(x)

if __name__ == "__main__":
    args = tyro.cli(Args)
    print(f"Initializing Phase 1 World Model (VAE + MDN-RNN) on {args.env_id}")
    
    # NOTE: Training a full World Model on CarRacing-v2 from scratch takes millions of frames
    # and hours/days on a single GPU. Here is the structured setup you would scale out:
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Components
    vae = VAE(args.latent_dim).to(device)
    rnn = MDNRNN(args.latent_dim, action_dim=3, hidden_dim=args.rnn_hidden_dim, num_mix=args.num_gaussian_mix).to(device)
    controller = Controller(args.latent_dim, args.rnn_hidden_dim, action_dim=3).to(device)
    
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
    rnn_optimizer = optim.Adam(rnn.parameters(), lr=args.rnn_lr)
    
    # In a full run you would:
    # 2. Collect 10,000 rollouts with a random policy
    print(f"[{time.strftime('%H:%M:%S')}] Step 1: Collect Random Rollouts (Skipped in stub)")
    
    # 3. Train VAE
    print(f"[{time.strftime('%H:%M:%S')}] Step 2: Train VAE to reconstruct frames (Skipped in stub)")
    # VAE Loss = MSE(reconstruction, original) + KLD(mu, sigma)
    
    # 4. Train MDN-RNN
    print(f"[{time.strftime('%H:%M:%S')}] Step 3: Train MDN-RNN forward dynamics (Skipped in stub)")
    # RNN Loss = NLL of Mixture of Gaussians prediction given z_t and a_t
    
    # 5. Train Controller via CMA-ES
    print(f"[{time.strftime('%H:%M:%S')}] Step 4: Evolution via CMA-ES in latent space")
    print("\nTo achieve the 'CarRacing solving' metric from David Ha's paper (>900 avg score),")
    print("the Controller is evolved by packing its weights into a flat vector using `cma.CMAEvolutionStrategy`")
    print("over parallelized environment rollouts purely observing `z_t` and `h_t`.")
    print("\nTraining configuration complete. Run the full data pipeline when compute is available.")

