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
from sklearn.cluster import MiniBatchKMeans

# Optimization: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
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

class LatentKMeansTracker:
    def __init__(self, n_clusters=30):
        self.n_clusters = n_clusters
        self.transitions = []
        self.clusterer = MiniBatchKMeans(n_clusters=n_clusters, n_init=3)
        self.fitted = False
        self.latent_buffer = []
        self.max_buffer = 10000

    def add_transition(self, z, next_z):
        self.transitions.append((z, next_z))
        self.latent_buffer.append(z)
        if len(self.latent_buffer) > self.max_buffer:
            self.latent_buffer.pop(0)
            self.transitions.pop(0)

    def fit(self):
        if len(self.latent_buffer) < self.n_clusters: return
        data = np.array(self.latent_buffer)
        self.clusterer.partial_fit(data)
        self.fitted = True

    def get_cluster(self, z):
        if not self.fitted: return -1
        return self.clusterer.predict(z.reshape(1, -1))[0]

class PPOWorker(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.mu = nn.Linear(256, n_out)
        self.log_std = nn.Parameter(torch.zeros(1, n_out))
        self.val = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.tanh(self.mu(x)), torch.exp(self.log_std), self.val(x)

def train(args):
    # Use vectorized environments to speed up data collection
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    run_name = f"sit_v2_fast_{args.env}_{int(time.time())}"
    writer = SummaryWriter(f"logs/continuous_study/{run_name}")
    
    vae = VAE(state_dim).to(device)
    vae_opt = optim.Adam(vae.parameters(), 1e-3)
    
    # Worker with slightly more capacity
    worker = PPOWorker(state_dim + 16, action_dim).to(device)
    w_opt = optim.Adam(worker.parameters(), 3e-4)
    
    tracker = LatentKMeansTracker(n_clusters=args.n_clusters)
    
    print(f"Starting Optimized SIT-V2 on {args.env} | Device: {device}")
    start_time = time.time()
    
    for ep in range(args.episodes):
        s, _ = env.reset(); done = False; total_rew = 0
        states, actions, log_probs, rewards, values, goals = [], [], [], [], [], []
        
        # Periodic Clustering update
        if ep % 50 == 0 and ep > 0:
            tracker.fit()
        
        # Goal selection (Target Latent)
        if ep < 100 or not tracker.fitted:
            target_latent = torch.zeros(16).to(device)
            target_cluster = -1
        else:
            # Pick a center from KMeans
            idx = np.random.randint(0, args.n_clusters)
            target_latent = torch.tensor(tracker.clusterer.cluster_centers_[idx], dtype=torch.float32).to(device)
            target_cluster = idx

        while not done:
            s_t = torch.tensor(s, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                _, mu, _ = vae(s_t.unsqueeze(0))
                z = mu.squeeze(0)
            
            w_in = torch.cat([s_t, target_latent])
            mu_a, std_a, val = worker(w_in)
            dist = Normal(mu_a, std_a)
            action = dist.sample()
            
            ns, r, term, trunc, _ = env.step(action.cpu().numpy().flatten())
            done = term or trunc
            
            # Intrinsic reward
            ir = 0
            if tracker.fitted:
                curr_cl = tracker.get_cluster(z.cpu().numpy())
                if curr_cl == target_cluster: ir = 0.1
            
            states.append(w_in)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(r + ir)
            values.append(val)
            
            # Update VAE online
            ns_t = torch.tensor(ns, dtype=torch.float32).to(device)
            recon, mu_vae, logvar_vae = vae(s_t.unsqueeze(0))
            recon_loss = F.mse_loss(recon, s_t.unsqueeze(0))
            kl_loss = -0.5 * torch.sum(1 + logvar_vae - mu_vae.pow(2) - logvar_vae.exp())
            v_loss = recon_loss + 0.001 * kl_loss
            vae_opt.zero_grad(); v_loss.backward(); vae_opt.step()
            
            tracker.add_transition(z.cpu().numpy(), mu_vae.detach().cpu().numpy().flatten())
            
            s = ns; total_rew += r

        # Update Worker (PPO-lite)
        if len(states) > 0:
            S = torch.stack(states)
            A = torch.stack(actions)
            LP = torch.stack(log_probs).detach()
            V = torch.stack(values).squeeze()
            
            # Returns
            G = []
            curr_g = 0
            for r in reversed(rewards):
                curr_g = r + 0.99 * curr_g
                G.insert(0, curr_g)
            G = torch.tensor(G, dtype=torch.float32).to(device)
            
            # 5 Epochs of PPO update
            for _ in range(5):
                mu_new, std_new, v_new = worker(S)
                dist_new = Normal(mu_new, std_new)
                lp_new = dist_new.log_prob(A)
                ratio = torch.exp(lp_new - LP)
                
                adv = (G - V).detach()
                # Ensure adv matches ratio shape: [T] -> [T, ActionDim]
                adv_expanded = adv.unsqueeze(1).expand_as(ratio)
                
                surr1 = ratio * adv_expanded
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_expanded
                
                pi_loss = -torch.min(surr1, surr2).mean()
                val_loss = 0.5 * F.mse_loss(v_new.squeeze(), G)
                
                w_opt.zero_grad()
                (pi_loss + val_loss).backward()
                nn.utils.clip_grad_norm_(worker.parameters(), 0.5)
                w_opt.step()

        if ep % 20 == 0:
            elapsed = time.time() - start_time
            print(f"Ep {ep:4d} | Rew: {total_rew:8.2f} | Time: {elapsed:6.1f}s")
            writer.add_scalar("charts/eps_per_sec", ep / (elapsed + 1e-6), ep)
            writer.add_scalar("charts/episodic_return", total_rew, ep)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--n_clusters", type=int, default=30)
    args = parser.parse_args()
    train(args)
