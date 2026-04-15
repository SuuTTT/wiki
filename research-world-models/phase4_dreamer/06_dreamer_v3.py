import os
import time
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../learn-cleanrl")))
from cleanrl_utils.logger import RLTracker
import gym

# ---------------------------------------------------------
# DreamerV3 Key Innovations (Symlog, Two-Hot, LayerNorm)
# ---------------------------------------------------------

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class TwoHotEncoding(nn.Module):
    def __init__(self, min_val=-20.0, max_val=20.0, bins=255):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.bins = bins
        self.step_size = (max_val - min_val) / (bins - 1)
        self.register_buffer("bucket_values", torch.linspace(min_val, max_val, bins))

    def encode(self, x):
        x = torch.clamp(x, self.min_val, self.max_val)
        scaled = (x - self.min_val) / self.step_size
        lower = torch.floor(scaled)
        upper = torch.ceil(scaled)
        upper_weight = scaled - lower
        lower_weight = 1.0 - upper_weight
        
        flat_x = x.view(-1, 1)
        flat_lower = torch.floor((flat_x - self.min_val) / self.step_size)
        flat_upper = torch.ceil((flat_x - self.min_val) / self.step_size)
        flat_upper_weight = ((flat_x - self.min_val) / self.step_size) - flat_lower
        flat_lower_weight = 1.0 - flat_upper_weight
        
        batch_size = flat_x.shape[0]
        two_hot = torch.zeros((batch_size, self.bins), device=x.device)
        two_hot.scatter_add_(-1, flat_lower.long(), flat_lower_weight)
        two_hot.scatter_add_(-1, flat_upper.long(), flat_upper_weight)
        
        target_shape = list(x.shape)
        target_shape[-1] = self.bins
        return two_hot.view(*target_shape)

    def decode(self, logits):
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.bucket_values, dim=-1, keepdim=True)

class LayerNormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        return self.norm(self.linear(x))

# ---------------------------------------------------------
# Real Discrete RSSM (DreamerV2/V3) 
# ---------------------------------------------------------

def st_sample(logits):
    """ Straight-through estimator for Categorical distributions """
    dist = torch.distributions.OneHotCategorical(logits=logits)
    sample = dist.sample()
    sample = sample + dist.probs - dist.probs.detach()
    return sample

class DiscreteRSSMV3(nn.Module):
    """
    State Space: h_t (deterministic GRU), z_t (Discrete Stochastic)
    z_t is a matrix of [32 Categories, 32 Classes]
    """
    def __init__(self, action_dim, embed_dim=256, h_dim=256, z_categories=32, z_classes=32):
        super().__init__()
        self.h_dim = h_dim
        self.z_categories = z_categories
        self.z_classes = z_classes
        self.z_dim = z_categories * z_classes
        
        # Recurrent deterministic core
        self.gru = nn.GRUCell(self.z_dim + action_dim, h_dim)
        
        # Prior (P(z_t | h_t))
        self.prior_mlp = nn.Sequential(
            LayerNormLinear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, self.z_dim)
        )
        
        # Posterior (Q(z_t | h_t, embed_t))
        self.posterior_mlp = nn.Sequential(
            LayerNormLinear(h_dim + embed_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, self.z_dim)
        )

    def step_prior(self, h_prev, z_prev, action):
        h_t = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)
        prior_logits = self.prior_mlp(h_t).view(-1, self.z_categories, self.z_classes)
        return h_t, prior_logits

    def step_posterior(self, h_t, embed):
        post_logits = self.posterior_mlp(torch.cat([h_t, embed], dim=-1)).view(-1, self.z_categories, self.z_classes)
        z_t = st_sample(post_logits).view(-1, self.z_dim)
        return post_logits, z_t

# ---------------------------------------------------------
# Environment Models (Reward, Decoder, Actor, Critic)
# ---------------------------------------------------------

class EncoderV3(nn.Module):
    def __init__(self, obs_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            LayerNormLinear(obs_dim, embed_dim), nn.SiLU(),
            LayerNormLinear(embed_dim, embed_dim), nn.SiLU()
        )
    def forward(self, obs):
        return self.net(symlog(obs))

class DecoderV3(nn.Module):
    def __init__(self, h_dim, z_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            LayerNormLinear(h_dim + z_dim, 256), nn.SiLU(),
            LayerNormLinear(256, 256), nn.SiLU(),
            nn.Linear(256, obs_dim)
        )
    def loss(self, h_t, z_t, obs):
        pred_symlog = self.net(torch.cat([h_t, z_t], dim=-1))
        # Reconstruction evaluated in symlog space safely
        # V3 strictly relies on MSE without variance tuning
        return F.mse_loss(pred_symlog, symlog(obs))

class RewardPredictorV3(nn.Module):
    def __init__(self, h_dim, z_dim, bins=255):
        super().__init__()
        self.net = nn.Sequential(
            LayerNormLinear(h_dim + z_dim, 256), nn.SiLU(),
            LayerNormLinear(256, 256), nn.SiLU(),
            nn.Linear(256, bins)
        )
        self.two_hot = TwoHotEncoding(bins=bins)
        
    def loss(self, h_t, z_t, reward_target):
        logits = self.net(torch.cat([h_t, z_t], dim=-1))
        target_symlog = symlog(reward_target)
        target_two_hot = self.two_hot.encode(target_symlog)
        return -torch.sum(target_two_hot * F.log_softmax(logits, dim=-1), dim=-1).mean()
        
    def predict(self, h_t, z_t):
        logits = self.net(torch.cat([h_t, z_t], dim=-1))
        symlog_val = self.two_hot.decode(logits)
        return symexp(symlog_val)

class ActorV3(nn.Module):
    def __init__(self, h_dim, z_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            LayerNormLinear(h_dim + z_dim, 256), nn.SiLU(),
            LayerNormLinear(256, 256), nn.SiLU(),
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.std_layer = nn.Linear(256, action_dim)

    def forward(self, h_t, z_t):
        x = self.net(torch.cat([h_t, z_t], dim=-1))
        mean = self.mean_layer(x)
        std = F.softplus(self.std_layer(x)) + 1e-4
        # Limit explosion
        dist = torch.distributions.Normal(torch.tanh(mean)*2.0, std) 
        return dist

class CriticV3(nn.Module):
    def __init__(self, h_dim, z_dim, bins=255):
        super().__init__()
        self.net = nn.Sequential(
            LayerNormLinear(h_dim + z_dim, 256), nn.SiLU(),
            LayerNormLinear(256, 256), nn.SiLU(),
            nn.Linear(256, bins)
        )
        self.two_hot = TwoHotEncoding(bins=bins)
        
    def get_value(self, h_t, z_t):
        logits = self.net(torch.cat([h_t, z_t], dim=-1))
        symlog_val = self.two_hot.decode(logits)
        return symexp(symlog_val)
        
    def loss(self, h_t, z_t, target_returns):
        target_symlog = symlog(target_returns)
        target_two_hot = self.two_hot.encode(target_symlog)
        logits = self.net(torch.cat([h_t, z_t], dim=-1))
        return -torch.sum(target_two_hot * F.log_softmax(logits, dim=-1), dim=-1).mean()

# ---------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=100000):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.capacity = capacity

    def add(self, obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch_sequence(self, batch_size, seq_len=50):
        # We need continuous sequences of length 'seq_len'
        starts = np.random.randint(0, self.size - seq_len, size=batch_size)
        
        obs_seq = np.zeros((seq_len, batch_size, self.obs.shape[-1]))
        act_seq = np.zeros((seq_len, batch_size, self.action.shape[-1]))
        rew_seq = np.zeros((seq_len, batch_size, 1))
        done_seq = np.zeros((seq_len, batch_size, 1))
        
        for i, start in enumerate(starts):
            obs_seq[:, i] = self.obs[start:start+seq_len]
            act_seq[:, i] = self.action[start:start+seq_len]
            rew_seq[:, i] = self.reward[start:start+seq_len]
            done_seq[:, i] = self.done[start:start+seq_len]
            
        return (torch.FloatTensor(obs_seq), torch.FloatTensor(act_seq),
                torch.FloatTensor(rew_seq), torch.FloatTensor(done_seq))

# ---------------------------------------------------------
# KL Divergence Balancing (V2 & V3 Feature)
# ---------------------------------------------------------
def kl_balancing_loss(prior_logits, post_logits, alpha=0.8):
    """ Keep Posterior sharp but force Prior to catch up smoothly (DreamerV2/V3) """
    prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
    post_dist = torch.distributions.OneHotCategorical(logits=post_logits)
    
    prior_dist_detached = torch.distributions.OneHotCategorical(logits=prior_logits.detach())
    post_dist_detached = torch.distributions.OneHotCategorical(logits=post_logits.detach())
    
    # Push Prior towards detached Posterior (80%)
    kl_prior = torch.distributions.kl.kl_divergence(post_dist_detached, prior_dist).sum(-1).mean()
    # Pull Posterior towards detached Prior (20%) - Regularization
    kl_post = torch.distributions.kl.kl_divergence(post_dist, prior_dist_detached).sum(-1).mean()
    
    # Free bits: Prevent KL from falling below 1.0 (prevents posterior collapse)
    return torch.max(kl_prior, torch.tensor(1.0, device=kl_prior.device)) * alpha + \
           torch.max(kl_post, torch.tensor(1.0, device=kl_post.device)) * (1 - alpha)

def compute_lambda_returns(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Computes true lambda returns flowing backwards through time.
    rewards: [H, B, 1]
    values: [H, B, 1] 
    returns: [H, B, 1]
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


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Real DreamerV3 on {device} (Pendulum-v1)...")
    
    os.makedirs("benchmark/06_dreamer_v3", exist_ok=True)
    tracker = RLTracker("06_dreamer_v3", 1)

    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Model Components
    encoder = EncoderV3(obs_dim, embed_dim=256).to(device)
    rssm = DiscreteRSSMV3(action_dim, embed_dim=256, h_dim=256, z_categories=32, z_classes=32).to(device)
    decoder = DecoderV3(256, 32*32, obs_dim).to(device)
    reward_pred = RewardPredictorV3(256, 32*32, bins=255).to(device)
    actor = ActorV3(256, 32*32, action_dim).to(device)
    critic = CriticV3(256, 32*32, bins=255).to(device)
    
    # Optimizers
    model_params = list(encoder.parameters()) + list(rssm.parameters()) + \
                   list(decoder.parameters()) + list(reward_pred.parameters())
    opt_model = torch.optim.Adam(model_params, lr=3e-4)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)
    
    buffer = ReplayBuffer(obs_dim, action_dim)
    
    # Warmup Buffer
    print("Warming up buffer...")
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        next_obs, reward, done, _, _ = env.step(action)
        buffer.add(obs, action, reward, done)
        obs = next_obs if not done else env.reset()[0]
        
    print("Starting Training Loop...")
    global_step = 1000
    ep_return = 0
    ep_len = 0
    obs, _ = env.reset()
    
    h_t = torch.zeros(1, 256, device=device)
    z_t = torch.zeros(1, 32*32, device=device)
    prev_act_env = torch.zeros(1, action_dim, device=device)
    
    while global_step < 1_000_000:
        with torch.no_grad():
            encoded_obs = encoder(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            h_t, _ = rssm.step_prior(h_t, z_t, prev_act_env)
            _, z_t = rssm.step_posterior(h_t, encoded_obs)
            
            action_dist = actor(h_t, z_t)
            # Add normal exploration noise or sample
            action = action_dist.sample().clamp(-2.0, 2.0).squeeze(0).cpu().numpy()
            
        next_obs, reward, done, trunc, _ = env.step(action)
        buffer.add(obs, action, reward, done)
        
        ep_return += reward
        ep_len += 1
        obs = next_obs
        prev_act_env = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        
        global_step += 1
        tracker.step(1)
        
        if done or trunc or ep_len >= 200:
            print(f"Step {global_step} | Return: {ep_return:.2f}")
            tracker.log_episode(ep_return, ep_len)
            obs, _ = env.reset()
            h_t = torch.zeros(1, 256, device=device)
            z_t = torch.zeros(1, 32*32, device=device)
            prev_act_env = torch.zeros(1, action_dim, device=device)
            ep_return = 0
            ep_len = 0

        # Train
        if global_step % 1 == 0:
            batch_obs, batch_act, batch_rew, batch_done = buffer.sample_batch_sequence(batch_size=32, seq_len=50)
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            batch_rew = batch_rew.to(device)
            
            h_train = torch.zeros(32, 256, device=device)
            z_train = torch.zeros(32, 32*32, device=device)
            
            loss_decoder, loss_reward, loss_kl = 0, 0, 0
            encoded_obs_train = encoder(batch_obs)
            
            posteriors_z = []
            histories_h = []
            
            prev_act = torch.zeros(32, action_dim, device=device)
            for t in range(50):
                h_train, prior_logits = rssm.step_prior(h_train, z_train, prev_act)
                post_logits, z_train = rssm.step_posterior(h_train, encoded_obs_train[t])
                
                posteriors_z.append(z_train)
                histories_h.append(h_train)
                
                loss_kl += kl_balancing_loss(prior_logits, post_logits)
                loss_decoder += decoder.loss(h_train, z_train, batch_obs[t])
                loss_reward += reward_pred.loss(h_train, z_train, batch_rew[t])
                
                prev_act = batch_act[t]
                
            loss_model = (loss_kl / 50) + (loss_decoder / 50) + (loss_reward / 50)
            opt_model.zero_grad()
            loss_model.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1000.0)
            opt_model.step()
            
            flat_h = torch.cat(histories_h, dim=0).detach() 
            flat_z = torch.cat(posteriors_z, dim=0).detach()
            
            imagined_h = flat_h.contiguous()
            imagined_z = flat_z.contiguous()
            
            horizon_h = []
            horizon_z = []
            horizon_rewards = []
            
            horizon_h.append(imagined_h)
            horizon_z.append(imagined_z)
            
            for _ in range(15):
                imagined_action = actor(imagined_h, imagined_z).rsample()
                imagined_h, prior_logits = rssm.step_prior(imagined_h, imagined_z, imagined_action)
                imagined_z = st_sample(prior_logits).view(-1, 32*32) 
                
                pred_reward = reward_pred.predict(imagined_h, imagined_z)
                
                horizon_h.append(imagined_h)
                horizon_z.append(imagined_z)
                horizon_rewards.append(pred_reward)
                
            stack_h = torch.stack(horizon_h)
            stack_z = torch.stack(horizon_z)
            imagined_values = critic.get_value(stack_h, stack_z).squeeze(-1) 
            stack_rew = torch.stack(horizon_rewards).squeeze(-1)
            
            target_returns = compute_lambda_returns(stack_rew, imagined_values[1:].detach(), gamma=0.99, lambda_=0.95).unsqueeze(-1)
            
            loss_actor = -target_returns.mean()
            
            for p in critic.parameters():
                p.requires_grad = False
                
            opt_actor.zero_grad()
            loss_actor.backward()
            opt_actor.step()
            
            for p in critic.parameters():
                p.requires_grad = True

            loss_critic = critic.loss(stack_h[1:].detach(), stack_z[1:].detach(), target_returns.detach())
            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

            if global_step % 500 == 0:
                print(f"Step {global_step} | Model: {loss_model.item():.2f} | Actor: {loss_actor.item():.2f} | Critic: {loss_critic.item():.2f} | KL: {(loss_kl/50).item():.2f}")
                tracker.log_metrics("losses", {
                    "model": loss_model.item(),
                    "actor": loss_actor.item(),
                    "critic": loss_critic.item(),
                    "kl": (loss_kl/50).item()
                })

    tracker.close()
    print("DreamerV3 Real Algorithm End-to-End Run Completed!")
