import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.kl import kl_divergence
import tyro
import envpool
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
    total_steps: int = 1_000_000
    learning_rate_model: float = 6e-4
    learning_rate_actor: float = 8e-5
    learning_rate_value: float = 8e-5
    batch_size: int = 50
    seq_len: int = 50          # BPTT chunk sequence length
    
    # Architecture (RSSM)
    stoch_size: int = 30       
    deter_size: int = 200      
    hidden_size: int = 200     
    action_dim: int = 3        
    
    # Dreamer specific
    horizon: int = 15          # Imagination horizon H
    gamma: float = 0.99
    lambda_: float = 0.95      # Lambda return decay

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =====================================================================
# RECURENT STATE SPACE MODEL (RSSM) & CHAINS
# =====================================================================
class RSSM(nn.Module):
    def __init__(self, action_dim, stoch_size, deter_size, hidden_size):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        
        self.cell = nn.GRUCell(self.stoch_size + action_dim, self.deter_size)
        
        self.fc_prior1 = layer_init(nn.Linear(self.deter_size, hidden_size))
        self.fc_prior_mean = layer_init(nn.Linear(hidden_size, self.stoch_size))
        self.fc_prior_std = layer_init(nn.Linear(hidden_size, self.stoch_size))
        
        self.fc_post1 = layer_init(nn.Linear(self.deter_size + 4096, hidden_size))
        self.fc_post_mean = layer_init(nn.Linear(hidden_size, self.stoch_size))
        self.fc_post_std = layer_init(nn.Linear(hidden_size, self.stoch_size))

    def prior(self, h):
        hidden = F.elu(self.fc_prior1(h))
        mean = self.fc_prior_mean(hidden)
        std = F.softplus(self.fc_prior_std(hidden)) + 0.1
        return mean, std

    def posterior(self, h, embed):
        hidden = F.elu(self.fc_post1(torch.cat([h, embed], dim=-1)))
        mean = self.fc_post_mean(hidden)
        std = F.softplus(self.fc_post_std(hidden)) + 0.1
        return mean, std
        
    def step_forward(self, h_prev, s_prev, action):
        x = torch.cat([s_prev, action], dim=-1)
        return self.cell(x, h_prev)

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
            nn.Sigmoid()
        )
    def forward(self, features):
        x = self.fc(features).view(-1, 1024, 1, 1)
        return self.net(x)

class DenseModel(nn.Module):
    def __init__(self, feature_dim, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 200)), nn.ELU(),
            layer_init(nn.Linear(200, 200)), nn.ELU(),
            layer_init(nn.Linear(200, out_dim))
        )
    def forward(self, features): return self.net(features)

class ActionModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 200)), nn.ELU(),
            layer_init(nn.Linear(200, 200)), nn.ELU()
        )
        self.mean_layer = layer_init(nn.Linear(200, action_dim), std=0.01)
        self.std_layer = layer_init(nn.Linear(200, action_dim), std=0.01)

    def forward(self, features, deterministic=False):
        x = self.net(features)
        mean = self.mean_layer(x)
        if deterministic:
            return torch.tanh(mean)
        std = F.softplus(self.std_layer(x)) + 1e-1
        std = torch.clamp(std, min=1e-1, max=1.0)
        # PyTorch differentiable sampling with reparameterization
        dist = Normal(mean, std)
        transforms = [TanhTransform()]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample() # rsample() is key for gradients to flow backwards!
        return action

class Dreamer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder()
        self.rssm = RSSM(args.action_dim, args.stoch_size, args.deter_size, args.hidden_size)
        self.decoder = Decoder(args.stoch_size + args.deter_size)
        self.reward_model = DenseModel(args.stoch_size + args.deter_size)
        self.value_model = DenseModel(args.stoch_size + args.deter_size)
        self.actor_model = ActionModel(args.stoch_size + args.deter_size, args.action_dim)

# =====================================================================
# LAMBDA RETURNS COMPUTATION
# =====================================================================
def compute_lambda_returns(rewards, values, gamma, lambda_):
    """
    Computes expanding horizon returns. 
    rewards: (H, B)
    values: (H, B)
    returns: (H, B) 
    """
    returns = torch.zeros_like(rewards)
    last_val = values[-1]
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            ret = rewards[t] + gamma * values[t]
        else:
            ret = rewards[t] + gamma * ((1 - lambda_) * values[t] + lambda_ * last_val)
        returns[t] = ret
        last_val = ret
    return returns

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    print(f"Initializing DreamerV1 on {args.env_id}")
    env = envpool.make(args.env_id, env_type="gymnasium", num_envs=1)
    tracker = RLTracker(args.exp_name, args.seed)
    
    model = Dreamer(args).to(device)
    
    # Split optimizers: Model vs. Actor vs. Value
    model_params = list(model.encoder.parameters()) + list(model.rssm.parameters()) + \
                   list(model.decoder.parameters()) + list(model.reward_model.parameters())
    optim_model = optim.Adam(model_params, lr=args.learning_rate_model)
    optim_actor = optim.Adam(model.actor_model.parameters(), lr=args.learning_rate_actor)
    optim_value = optim.Adam(model.value_model.parameters(), lr=args.learning_rate_value)
    
    max_steps = 10000 
    obs_b = torch.zeros((max_steps, 3, 64, 64))
    act_b = torch.zeros((max_steps, args.action_dim))
    rew_b = torch.zeros((max_steps, 1))
    
    def process_obs(obs):
        obs = torch.tensor(obs, dtype=torch.float32) / 255.0
        obs = obs.permute(0, 3, 1, 2)
        return F.interpolate(obs, size=(64, 64))

    print("Pre-filling buffer...")
    obs, _ = env.reset()
    for t in range(500):
        action = torch.rand(1, args.action_dim) * 2 - 1
        obs_b[t] = process_obs(obs)[0]
        act_b[t] = action[0]
        next_obs, reward, term, trunc, _ = env.step(action.numpy())
        rew_b[t] = float(reward[0])
        obs = next_obs if not (term[0] or trunc[0]) else env.reset()[0]

    # Main Loop
    global_step = 500
    obs, _ = env.reset()
    h_t = torch.zeros((1, args.deter_size)).to(device)
    s_t = torch.zeros((1, args.stoch_size)).to(device)
    ep_return = 0
    ep_len = 0
    
    for global_step in range(500, args.total_steps):
        # 1. ENVIRONMENT STEP: Act via Actor Network (NOT CEM!)
        with torch.no_grad():
            img_embed = model.encoder(process_obs(obs).to(device))
            post_mean, post_std = model.rssm.posterior(h_t, img_embed)
            s_t = post_mean + post_std * torch.randn_like(post_mean)
            
            features = torch.cat([h_t, s_t], dim=-1)
            # Explore via sampling or act deterministically
            action = model.actor_model(features, deterministic=(global_step < 1000))
            h_t = model.rssm.step_forward(h_t, s_t, action)

        next_obs, reward, term, trunc, _ = env.step(action.cpu().numpy())
        
        idx = global_step % max_steps
        obs_b[idx] = process_obs(obs)[0]
        act_b[idx] = action[0].cpu()
        rew_b[idx] = float(reward[0])
        
        obs = next_obs
        ep_return += reward[0]
        ep_len += 1
        
        if term[0] or trunc[0]:
            print(f"Step {global_step} | Return: {ep_return:.2f}")
            tracker.log_episode(ep_return, ep_len)
            obs, _ = env.reset()
            h_t = torch.zeros((1, args.deter_size)).to(device)
            s_t = torch.zeros((1, args.stoch_size)).to(device)
            ep_return, ep_len = 0, 0
            
        tracker.step(1)
        
        # 2. TRAIN STEP
        if global_step % 50 == 0:
            # Sample continuous chunks
            idx = global_step % max_steps
            starts = []
            while len(starts) < args.batch_size:
                s = np.random.randint(0, min(global_step, max_steps) - args.seq_len)
                # Avoid sequences that cross the current overwrite pointer
                if global_step >= max_steps and (s <= idx < s + args.seq_len):
                    continue
                starts.append(s)
            starts = np.array(starts)
            batch_obs = torch.stack([obs_b[s:s+args.seq_len] for s in starts]).to(device) 
            batch_act = torch.stack([act_b[s:s+args.seq_len] for s in starts]).to(device)
            batch_rew = torch.stack([rew_b[s:s+args.seq_len] for s in starts]).to(device)
            
            # ================== TRAIN WORLD MODEL ==================
            h = torch.zeros(args.batch_size, args.deter_size).to(device)
            s = torch.zeros(args.batch_size, args.stoch_size).to(device)
            
            kl_loss, recon_loss, reward_loss = 0, 0, 0
            # Collect posteriors to start imagination from
            post_states = []
            post_hiddens = []
            
            prev_act = torch.zeros(args.batch_size, args.action_dim).to(device)
            
            for t in range(args.seq_len):
                # Step forward uses PREVIOUS action to predict CURRENT state dynamics
                h = model.rssm.step_forward(h, s, prev_act)
                embed = model.encoder(batch_obs[:, t])
                post_mean, post_std = model.rssm.posterior(h, embed)
                post_dist = Normal(post_mean, post_std)
                s = post_dist.rsample()
                
                prior_mean, prior_std = model.rssm.prior(h)
                prior_dist = Normal(prior_mean, prior_std)
                
                features = torch.cat([h, s], dim=-1)
                kl_loss += kl_divergence(post_dist, prior_dist).sum(-1).mean()
                recon_loss += F.mse_loss(model.decoder(features), batch_obs[:, t])
                reward_loss += F.mse_loss(model.reward_model(features), batch_rew[:, t])
                
                # Detach for Actor training starting points so gradients don't flow back into model training here
                post_states.append(s.detach())
                post_hiddens.append(h.detach())
                prev_act = batch_act[:, t]
                
            model_loss = (kl_loss + recon_loss + reward_loss) / args.seq_len
            optim_model.zero_grad()
            model_loss.backward()
            nn.utils.clip_grad_norm_(model_params, 100.0)
            optim_model.step()
            
            # ================== TRAIN ACTOR & CRITIC ==================
            # Flatten (Seq, Batch, Dim) -> (Seq * Batch, Dim)
            h_imag = torch.stack(post_hiddens).reshape(-1, args.deter_size)
            s_imag = torch.stack(post_states).reshape(-1, args.stoch_size)
            
            imag_states = []
            imag_hiddens = []
            imag_rewards = []
            
            # Rollout in Imagination
            for t in range(args.horizon):
                features = torch.cat([h_imag, s_imag], dim=-1)
                action = model.actor_model(features) # gradients flow through action
                h_imag = model.rssm.step_forward(h_imag, s_imag, action)
                prior_mean, prior_std = model.rssm.prior(h_imag)
                s_imag = prior_mean + prior_std * torch.randn_like(prior_mean)
                
                feat_imag = torch.cat([h_imag, s_imag], dim=-1)
                imag_rewards.append(model.reward_model(feat_imag).squeeze(-1))
                imag_states.append(s_imag)
                imag_hiddens.append(h_imag)
                
            imag_rewards = torch.stack(imag_rewards) # (H, Batch)
            imag_features = torch.cat([torch.stack(imag_hiddens), torch.stack(imag_states)], dim=-1)
            imag_values = model.value_model(imag_features).squeeze(-1) # (H, Batch)
            
            # Compute targets
            returns = compute_lambda_returns(imag_rewards, imag_values, args.gamma, args.lambda_)
            
            # Actor tries to maximize the return prediction
            actor_loss = -returns.mean()
            optim_actor.zero_grad()
            actor_loss.backward(retain_graph=True) # Gradients flow through RSSM backwards to Actor weights
            nn.utils.clip_grad_norm_(model.actor_model.parameters(), 100.0)
            optim_actor.step()
            
            # Value tries to predict the returns (Must recalculate from features without gradients affecting Actor update)
            # Recreate detached values for critic training to avoid inplace tensor version mismatch
            imag_features_detached = imag_features.detach()
            imag_values_critic = model.value_model(imag_features_detached).squeeze(-1)
            
            val_loss = F.mse_loss(imag_values_critic, returns.detach())
            optim_value.zero_grad()
            val_loss.backward()
            nn.utils.clip_grad_norm_(model.value_model.parameters(), 100.0)
            optim_value.step()
            
            tracker.log_metrics("losses", {
                "kl": (kl_loss/args.seq_len).item(),
                "recon": (recon_loss/args.seq_len).item(),
                "reward": (reward_loss/args.seq_len).item(),
                "actor": actor_loss.item(),
                "value": val_loss.item()
            })
            tracker.log_sps()
            
    tracker.close()