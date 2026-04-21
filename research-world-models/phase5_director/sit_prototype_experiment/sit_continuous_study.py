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
import argparse
from collections import deque

class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
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
        self.clusters = {} 
        
    def quantize(self, z):
        q = np.round(z.detach().cpu().numpy() * self.grid_size) / self.grid_size
        return tuple(q.flatten())

    def update(self, z, next_z):
        self.transitions.append((self.quantize(z), self.quantize(next_z)))
        if len(self.transitions) > 10000: self.transitions.pop(0)

    def run_clustering(self):
        G = nx.Graph()
        for s, ns in self.transitions:
            if G.has_edge(s, ns): G[s][ns]['weight'] += 1
            else: G.add_edge(s, ns, weight=1)
        
        if G.number_of_edges() > 0:
            try:
                self.clusters = community_louvain.best_partition(G)
                return len(set(self.clusters.values()))
            except:
                return 1
        return 1

class ContinuousAC(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, n_out)
        self.sigma = nn.Linear(128, n_out)
        self.val = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.tanh(self.mu(x)), F.softplus(self.sigma(x)) + 1e-5, self.val(x)

def train(args):
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    run_name = f"sit_cont_{args.env}_{int(time.time())}"
    writer = SummaryWriter(f"logs/continuous_study/{run_name}")
    
    vae = VAE(state_dim)
    vae_opt = optim.Adam(vae.parameters(), 1e-3)
    
    worker = ContinuousAC(state_dim + 16, action_dim) 
    w_opt = optim.Adam(worker.parameters(), 3e-4) # Standard PPO-like LR
    
    graph = LatentGraphTracker(latent_dim=16)
    
    print(f"Starting Continuous SIT on {args.env} | Log: {run_name}")
    
    for ep in range(args.episodes):
        s, _ = env.reset(); done = False; total_rew = 0
        w_hist = []
        
        # Frozen SIT Logic for Continuous
        if ep < 50:
            target_cluster = -1
            target_latent = torch.zeros(16)
        else:
            # Simple goal: pick a frequent cluster from the last run_clustering
            seen_clusters = list(set(graph.clusters.values()))
            target_cluster = np.random.choice(seen_clusters) if seen_clusters else 0
            # Target latent is currently a zero vector for simplicity in prototype
            target_latent = torch.zeros(16) 

        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            
            # VAE Step
            recon, mu, logvar = vae(s_t)
            z = vae.reparameterize(mu, logvar)
            
            # Worker Action
            w_in = torch.cat([s_t, target_latent])
            mu_a, sig_a, val = worker(w_in)
            dist = Normal(mu_a, sig_a)
            action = dist.sample()
            
            # Action Clipping for Continuous Env
            act_clipped = torch.clamp(action, env.action_space.low[0], env.action_space.high[0]).detach().numpy()
            
            ns, r, term, trunc, _ = env.step(act_clipped)
            done = term or trunc
            ns_t = torch.tensor(ns, dtype=torch.float32)
            
            # Update Graph
            with torch.no_grad():
                _, n_mu, _ = vae(ns_t)
                graph.update(mu, n_mu)
            
            # VAE Loss
            recon_loss = F.mse_loss(recon, s_t)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + 0.001 * kl_loss
            vae_opt.zero_grad(); vae_loss.backward(); vae_opt.step()
            
            # Intrinsic Reward logic (Stay in target cluster)
            with torch.no_grad():
                curr_q = graph.quantize(mu)
                curr_cl = graph.clusters.get(curr_q, -2)
                ir = 0.05 if curr_cl == target_cluster else 0.0
            
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
            # Simple policy gradient update
            log_p_sum = torch.stack(log_probs).sum(dim=-1)
            w_loss = -(log_p_sum * adv).mean() + 0.5 * F.mse_loss(vals, R)
            w_opt.zero_grad(); w_loss.backward(); w_opt.step()

        if ep % 10 == 0:
            n_cl = graph.run_clustering()
            print(f"[{args.env}] Ep {ep:4d} | Reward: {total_rew:8.2f} | Clusters: {n_cl:4d}")
            writer.add_scalar("charts/episodic_return", total_rew, ep)
            writer.add_scalar("SIT/clusters", n_cl, ep)
            writer.add_scalar("losses/vae_loss", vae_loss.item(), ep)

    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    train(args)
