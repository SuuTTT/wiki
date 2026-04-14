import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import tyro
import envpool
import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../learn-cleanrl")))
from cleanrl_utils.logger import RLTracker

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True
    env_id: str = "CarRacing-v2"
    
    # Training Parameters
    total_steps: int = 100_000 # Shortened for testing
    learning_rate: float = 1e-3
    batch_size: int = 32
    seq_len: int = 32          # BPTT chunk sequence length
    
    # Architecture (PlaNet/RSSM)
    stoch_size: int = 30       # Latent Gaussian dimensions
    deter_size: int = 200      # GRU Hidden State
    hidden_size: int = 200     # Dense mapping layers
    
    # Planning (CEM)
    planning_horizon: int = 12 # H: Action sequences to imagine
    cem_iters: int = 10        # Optimization iterations
    cem_candidates: int = 1000 # N: Initial paths
    cem_top_k: int = 100       # K: Top paths to refit
    action_dim: int = 3        # Steer, Gas, Brake

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =====================================================================
# RECURENT STATE SPACE MODEL (RSSM)
# =====================================================================
class RSSM(nn.Module):
    def __init__(self, action_dim, stoch_size, deter_size, hidden_size):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        
        # 1. Action + Stochastic -> Deterministic (Memory Update)
        self.cell = nn.GRUCell(self.stoch_size + action_dim, self.deter_size)
        
        # 2. Deterministic -> Prior Stochastic (Imagination without Seeing)
        self.fc_prior1 = layer_init(nn.Linear(self.deter_size, hidden_size))
        self.fc_prior_mean = layer_init(nn.Linear(hidden_size, self.stoch_size))
        self.fc_prior_std = layer_init(nn.Linear(hidden_size, self.stoch_size))
        
        # 3. Deterministic + Image CNN Features -> Posterior Stochastic (True Belief)
        self.fc_post1 = layer_init(nn.Linear(self.deter_size + 4096, hidden_size)) # 256*4*4 from CNN = 4096
        self.fc_post_mean = layer_init(nn.Linear(hidden_size, self.stoch_size))
        self.fc_post_std = layer_init(nn.Linear(hidden_size, self.stoch_size))

    def prior(self, h):
        """Calculates p(s_t|h_t) using only the GRU memory."""
        hidden = F.relu(self.fc_prior1(h))
        mean = self.fc_prior_mean(hidden)
        std = F.softplus(self.fc_prior_std(hidden)) + 0.1
        return mean, std

    def posterior(self, h, embed):
        """Calculates q(s_t|h_t, x_t) using actual encoded pixel frame."""
        hidden = F.relu(self.fc_post1(torch.cat([h, embed], dim=-1)))
        mean = self.fc_post_mean(hidden)
        std = F.softplus(self.fc_post_std(hidden)) + 0.1
        return mean, std
        
    def step_forward(self, h_prev, s_prev, action):
        """Moves deterministic RNN forward."""
        x = torch.cat([s_prev, action], dim=-1)
        h_t = self.cell(x, h_prev)
        return h_t

# =====================================================================
# ENCODER, DECODER, & REWARD PREDICTOR
# =====================================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 4, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, 4, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = layer_init(nn.Linear(feature_dim, 1024))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid() # Output pixels [0, 1]
        )
    def forward(self, features):
        x = self.fc(features)
        # Reshape so spatial dimensions match the expected inverted flattening
        x = x.view(-1, 1024, 1, 1)
        return self.net(x)

class RewardPredictor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 200)), nn.ReLU(),
            layer_init(nn.Linear(200, 200)), nn.ReLU(),
            layer_init(nn.Linear(200, 1))
        )
    def forward(self, features): return self.net(features)

class PlaNetDynamics(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder()
        self.rssm = RSSM(args.action_dim, args.stoch_size, args.deter_size, args.hidden_size)
        self.decoder = Decoder(args.stoch_size + args.deter_size)
        self.reward_model = RewardPredictor(args.stoch_size + args.deter_size)

# =====================================================================
# CROSS-ENTROPY METHOD (CEM) PLANNER 
# =====================================================================
def cem_plan(model, h_start, s_start, args, device):
    """
    Imagine args.planning_horizon into the future in Latent Space using purely the RSSM Prior.
    Returns the absolute best action out of `N` optimized imaginary tests.
    """
    # 1. Initialize random actions (mean 0, std 1)
    action_mean = torch.zeros((args.planning_horizon, args.action_dim)).to(device)
    action_std = torch.ones((args.planning_horizon, args.action_dim)).to(device)
    
    for i in range(args.cem_iters):
        # Sample N candidate sequences: (H, N, A)
        actions = Normal(action_mean, action_std).rsample([args.cem_candidates]).transpose(0, 1)
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Expand starting states to simulate N parallel futures
        h_t = h_start.expand(args.cem_candidates, -1)
        s_t = s_start.expand(args.cem_candidates, -1)
        returns = torch.zeros(args.cem_candidates).to(device)
        
        # Unroll the sequence in imagination using ONLY the RSSM Prior
        for t in range(args.planning_horizon):
            h_t = model.rssm.step_forward(h_t, s_t, actions[t])
            prior_mean, prior_std = model.rssm.prior(h_t)
            # Differentiable sample 
            s_t = prior_mean + prior_std * torch.randn_like(prior_mean) 
            
            features = torch.cat([h_t, s_t], dim=-1)
            predicted_reward = model.reward_model(features).flatten()
            returns += predicted_reward
            
        # Select best K return paths
        _, topk_indices = torch.topk(returns, args.cem_top_k)
        best_actions = actions[:, topk_indices, :] # (H, K, A)
        
        # Refit Gaussian distribution to the top K imaginary performing sequences
        action_mean = best_actions.mean(dim=1)
        action_std = best_actions.std(dim=1)
        
    # Return the FIRST action of the BEST found path
    best_action = action_mean[0] # Execution
    return best_action

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    print(f"Initializing PlaNet Reimplementation on {args.env_id}")
    env = envpool.make(args.env_id, env_type="gymnasium", num_envs=1)
    tracker = RLTracker(args.exp_name, args.seed)
    
    model = PlaNetDynamics(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # -----------------------------------------------------
    # Dummy Replay Buffer (for script viability/CleanRL layout)
    # -----------------------------------------------------
    max_steps = 5000 
    obs_buffer = torch.zeros((max_steps, 3, 64, 64))
    action_buffer = torch.zeros((max_steps, args.action_dim))
    reward_buffer = torch.zeros((max_steps, 1))
    
    def process_obs(obs):
        obs = torch.tensor(obs, dtype=torch.float32) / 255.0
        obs = obs.permute(0, 3, 1, 2)
        return F.interpolate(obs, size=(64, 64))

    # Fill Buffer with dummy random rollouts initially
    print("Pre-filling buffer...")
    obs, info = env.reset()
    for t in range(500):
        action = torch.rand(1, args.action_dim) * 2 - 1
        obs_buffer[t] = process_obs(obs)[0]
        action_buffer[t] = action[0]
        next_obs, reward, term, trunc, info = env.step(action.numpy())
        reward_buffer[t] = float(reward[0])
        obs = next_obs if not (term[0] or trunc[0]) else env.reset()[0]

    # -----------------------------------------------------
    # Main Training & Control Loop
    # -----------------------------------------------------
    global_step = 500
    obs, info = env.reset()
    
    # Trackers for rolling memory
    h_t = torch.zeros((1, args.deter_size)).to(device)
    s_t = torch.zeros((1, args.stoch_size)).to(device)
    
    ep_return = 0
    ep_len = 0
    
    for step in range(args.total_steps):
        # 1. ENVIRONMENT STEP: Act via CEM Planner
        with torch.no_grad():
            img_embed = model.encoder(process_obs(obs).to(device))
            post_mean, post_std = model.rssm.posterior(h_t, img_embed)
            s_t = post_mean + post_std * torch.randn_like(post_mean) # Update True Belief
            
            # Plan best action through latent overshooting
            if global_step < 1000: # Warmup
                action = torch.rand(1, args.action_dim).to(device) * 2 - 1
            else:
                action = cem_plan(model, h_t, s_t, args, device).unsqueeze(0)
            
            # Update memory sequence for next frame
            h_t = model.rssm.step_forward(h_t, s_t, action)

        next_obs, reward, term, trunc, info = env.step(action.cpu().numpy())
        
        # 2. STORE TRANSITION
        idx = global_step % max_steps
        obs_buffer[idx] = process_obs(obs)[0]
        action_buffer[idx] = action[0].cpu()
        reward_buffer[idx] = float(reward[0])
        
        obs = next_obs
        global_step += 1
        ep_return += reward[0]
        ep_len += 1
        
        if term[0] or trunc[0]:
            print(f"Step {global_step} | Return: {ep_return:.2f}")
            tracker.log_episode(ep_return, ep_len)
            obs, info = env.reset()
            h_t = torch.zeros((1, args.deter_size)).to(device)
            s_t = torch.zeros((1, args.stoch_size)).to(device)
            ep_return = 0; ep_len = 0
            
        tracker.step(1)
        
        # 3. TRAIN RSSM ON BATCHES (Temporal Alignment)
        if global_step % 50 == 0 and global_step > 1000:
            # Sample continuous chunks (B, T, C, H, W)
            starts = np.random.randint(0, min(global_step, max_steps) - args.seq_len, args.batch_size)
            batch_obs = torch.stack([obs_buffer[s:s+args.seq_len] for s in starts]).to(device) # (B, T, C, 64, 64)
            batch_act = torch.stack([action_buffer[s:s+args.seq_len] for s in starts]).to(device)
            batch_rew = torch.stack([reward_buffer[s:s+args.seq_len] for s in starts]).to(device)
            
            # Recurrent Pass Unroll
            h = torch.zeros(args.batch_size, args.deter_size).to(device)
            s = torch.zeros(args.batch_size, args.stoch_size).to(device)
            kl_loss_total = 0; recon_loss_total = 0; reward_loss_total = 0
            
            for t in range(args.seq_len):
                # Drive GRU
                h = model.rssm.step_forward(h, s, batch_act[:, t])
                
                # Image Encoder
                embed = model.encoder(batch_obs[:, t])
                
                # Posterior (See the Truth)
                post_mean, post_std = model.rssm.posterior(h, embed)
                post_dist = Normal(post_mean, post_std)
                s = post_dist.rsample()
                
                # Prior (Blind Imagination - KL Target)
                prior_mean, prior_std = model.rssm.prior(h)
                prior_dist = Normal(prior_mean, prior_std)
                
                # Losses
                features = torch.cat([h, s], dim=-1)
                
                kl_loss_total += kl_divergence(post_dist, prior_dist).sum(-1).mean()
                recon_loss_total += F.mse_loss(model.decoder(features), batch_obs[:, t])
                reward_loss_total += F.mse_loss(model.reward_model(features), batch_rew[:, t])
                
            loss = (kl_loss_total + recon_loss_total + reward_loss_total) / args.seq_len
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
            optimizer.step()
            
            tracker.log_metrics("losses", {
                "kl": (kl_loss_total/args.seq_len).item(),
                "recon": (recon_loss_total/args.seq_len).item(),
                "reward": (reward_loss_total/args.seq_len).item()
            })
            tracker.log_sps()
            
    tracker.close()