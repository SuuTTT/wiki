import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import community as community_louvain
import time
import os
from collections import deque

class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2) # mu and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class LatentGraphTracker:
    def __init__(self, latent_dim, grid_size=5):
        self.grid_size = grid_size
        self.transitions = []
        self.clusters = {} # z_quantized -> cluster_id
        
    def quantize(self, z):
        # Simple grid-based quantization for community detection
        # In a real setup, we might use k-means or VQ-VAE
        q = np.round(z.detach().numpy() * self.grid_size) / self.grid_size
        return tuple(q.flatten())

    def update(self, z, next_z):
        self.transitions.append((self.quantize(z), self.quantize(next_z)))
        if len(self.transitions) > 5000: self.transitions.pop(0)

    def run_clustering(self):
        G = nx.Graph()
        for s, ns in self.transitions:
            if G.has_edge(s, ns): G[s][ns]['weight'] += 1
            else: G.add_edge(s, ns, weight=1)
        
        if G.number_of_edges() > 0:
            self.clusters = community_louvain.best_partition(G)
            return len(set(self.clusters.values()))
        return 1

class ContinuousAC(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, 128)
        self.mu = nn.Linear(128, n_out)
        self.sigma = nn.Linear(128, n_out)
        self.val = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return torch.tanh(self.mu(x)), F.softplus(self.sigma(x)) + 1e-5, self.val(x)

def train_continuous_sit(env_name="Pendulum-v1", episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    vae = VAE(state_dim)
    vae_opt = optim.Adam(vae.parameters(), 1e-3)
    
    # Manager: picks latent cluster ID as goal
    # Worker: takes state + goal_latent -> action
    worker = ContinuousAC(state_dim + 16, action_dim) # 16 is latent_dim
    w_opt = optim.Adam(worker.parameters(), 1e-3)
    
    graph = LatentGraphTracker(latent_dim=16)
    writer = SummaryWriter(f"logs/continuous_sit_{int(time.time())}")
    
    print(f"Starting Continuous SIT on {env_name}...")
    
    for ep in range(episodes):
        s, _ = env.reset(); done = False; total_rew = 0
        w_hist = []
        
        # Simple goal selection for prototype: pick a random seen cluster
        target_cluster = 0
        target_latent = torch.zeros(16)
        
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            
            # VAE Step
            recon, mu, logvar = vae(s_t)
            z = vae.reparameterize(mu, logvar)
            
            # Action
            w_in = torch.cat([s_t, target_latent])
            mu_a, sig_a, val = worker(w_in)
            dist = Normal(mu_a, sig_a)
            action = dist.sample()
            
            ns, r, term, trunc, _ = env.step(action.numpy())
            done = term or trunc
            ns_t = torch.tensor(ns, dtype=torch.float32)
            
            # Update Graph
            with torch.no_grad():
                _, n_mu, _ = vae(ns_t)
                graph.update(mu, n_mu)
            
            # VAE Loss (online)
            recon_loss = F.mse_loss(recon, s_t)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + 0.01 * kl_loss
            vae_opt.zero_grad(); vae_loss.backward(); vae_opt.step()
            
            # Intrinsic Reward (if in same latent cluster)
            curr_cl = graph.clusters.get(graph.quantize(mu), -1)
            ir = 0.1 if curr_cl == target_cluster else -0.01
            
            w_hist.append((val, dist.log_prob(action), r + ir))
            s = ns; total_rew += r

        # Optimize Worker
        if len(w_hist) > 0:
            vals, log_probs, rews = zip(*w_hist)
            vals = torch.stack(vals).view(-1)
            R = []; curr_r = 0
            for r_val in reversed(rews):
                curr_r = r_val + 0.99 * curr_r
                R.insert(0, curr_r)
            R = torch.tensor(R, dtype=torch.float32)
            adv = R - vals.detach()
            w_loss = -(torch.stack(log_probs).sum(dim=-1) * adv).mean() + 0.5 * F.mse_loss(vals, R)
            w_opt.zero_grad(); w_loss.backward(); w_opt.step()

        if ep % 20 == 0:
            n_cl = graph.run_clustering()
            print(f"Ep {ep} | Reward: {total_rew:.2f} | Clusters: {n_cl}")
            writer.add_scalar("charts/return", total_rew, ep)
            writer.add_scalar("SIT/clusters", n_cl, ep)

    env.close()
    writer.close()

if __name__ == "__main__":
    train_continuous_sit()
