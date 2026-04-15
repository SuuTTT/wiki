import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import OneHotCategorical, Independent
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
    env_id: str = "Pong-v5" # DreamerV2 target domain is Atari
    
    # Training Parameters
    total_steps: int = 50_000_000
    learning_rate_model: float = 2e-4
    learning_rate_actor: float = 4e-5
    learning_rate_value: float = 1e-4
    batch_size: int = 50
    seq_len: int = 50          
    
    # Architecture (Discrete RSSM)
    category_size: int = 32    # 32 Categorical variables
    class_size: int = 32       # 32 Classes each
    deter_size: int = 200      
    hidden_size: int = 200     
    action_dim: int = 6        # Pong actions
    
    # DreamerV2 specific
    horizon: int = 15          
    gamma: float = 0.99
    lambda_: float = 0.95      
    kl_balance: float = 0.8    # Scale prior towards posterior faster than posterior towards prior
    free_nats: float = 0.1     # Minimum KL divergence threshold

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =====================================================================
# DISCRETE RECURENT STATE SPACE MODEL (RSSM)
# =====================================================================
class DiscreteRSSM(nn.Module):
    def __init__(self, action_dim, category_size, class_size, deter_size, hidden_size):
        super().__init__()
        self.cat_size = category_size
        self.cls_size = class_size
        self.stoch_size = category_size * class_size
        self.deter_size = deter_size
        
        # Action is one-hot (action_dim)
        self.cell = nn.GRUCell(self.stoch_size + action_dim, self.deter_size)
        
        self.fc_prior1 = layer_init(nn.Linear(self.deter_size, hidden_size))
        self.fc_prior_logits = layer_init(nn.Linear(hidden_size, self.stoch_size))
        
        # 4 channels * feature maps
        self.fc_post1 = layer_init(nn.Linear(self.deter_size + 4096, hidden_size))
        self.fc_post_logits = layer_init(nn.Linear(hidden_size, self.stoch_size))

    def prior(self, h):
        hidden = F.elu(self.fc_prior1(h))
        logits = self.fc_prior_logits(hidden).view(-1, self.cat_size, self.cls_size)
        return logits

    def posterior(self, h, embed):
        hidden = F.elu(self.fc_post1(torch.cat([h, embed], dim=-1)))
        logits = self.fc_post_logits(hidden).view(-1, self.cat_size, self.cls_size)
        return logits
        
    def step_forward(self, h_prev, s_prev, action):
        x = torch.cat([s_prev, action], dim=-1)
        return self.cell(x, h_prev)

def st_sample(logits):
    """ Straight-Through Gumbel-Softmax equivalent using OneHotCategorical """
    dist = OneHotCategorical(logits=logits)
    sample = dist.sample()
    # Sample gradient routing: forward pass is discrete sample, backward pass flows through probabilities
    return sample + dist.probs - dist.probs.detach()

# =====================================================================
# ENVPOOL ATARI ARCHITECTURE (4 Grayscale Channels)
# =====================================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 4, stride=2, padding=1)), nn.ELU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)), nn.ELU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, padding=1)), nn.ELU(),
            layer_init(nn.Conv2d(128, 256, 4, stride=2, padding=1)), nn.ELU(),
            nn.Flatten()
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = layer_init(nn.Linear(feature_dim, 1024))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ELU(),
            nn.ConvTranspose2d(32, 4, 6, stride=2),
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

class DiscreteActionModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 200)), nn.ELU(),
            layer_init(nn.Linear(200, 200)), nn.ELU(),
            layer_init(nn.Linear(200, action_dim), std=0.01)
        )

    def forward(self, features, deterministic=False):
        logits = self.net(features)
        if deterministic:
            idx = torch.argmax(logits, dim=-1)
            return F.one_hot(idx, num_classes=logits.shape[-1]).float()
        
        # We return a distribution here so we can compute log_probs easily for REINFORCE in DreamerV2
        dist = torch.distributions.OneHotCategorical(logits=logits)
        return dist

class DreamerV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        stoch_size = args.category_size * args.class_size
        self.encoder = Encoder()
        self.rssm = DiscreteRSSM(args.action_dim, args.category_size, args.class_size, args.deter_size, args.hidden_size)
        self.decoder = Decoder(stoch_size + args.deter_size)
        self.reward_model = DenseModel(stoch_size + args.deter_size)
        self.value_model = DenseModel(stoch_size + args.deter_size)
        self.actor_model = DiscreteActionModel(stoch_size + args.deter_size, args.action_dim)

# =====================================================================
# LAMBDA RETURNS COMPUTATION
# =====================================================================
def compute_lambda_returns(rewards, values, gamma, lambda_):
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

def calculate_kl_loss(post_logits, prior_logits, alpha, free_nats):
    """ KL Balancing from DreamerV2 Paper """
    post_dist = Independent(OneHotCategorical(logits=post_logits), 1)
    prior_dist = Independent(OneHotCategorical(logits=prior_logits), 1)
    
    post_detached_dist = Independent(OneHotCategorical(logits=post_logits.detach()), 1)
    prior_detached_dist = Independent(OneHotCategorical(logits=prior_logits.detach()), 1)

    # KL(post.detach() || prior)
    kl_lhs = torch.distributions.kl.kl_divergence(post_detached_dist, prior_dist)
    # KL(post || prior.detach())
    kl_rhs = torch.distributions.kl.kl_divergence(post_dist, prior_detached_dist)
    
    # Apply free nats (minimum threshold)
    kl_lhs = torch.max(kl_lhs, torch.ones_like(kl_lhs) * free_nats)
    kl_rhs = torch.max(kl_rhs, torch.ones_like(kl_rhs) * free_nats)
    
    return (alpha * kl_lhs + (1 - alpha) * kl_rhs).mean()

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    print(f"Initializing DreamerV2 on {args.env_id} (Discrete Controls, KL Balancing)")
    env = envpool.make(args.env_id, env_type="gymnasium", num_envs=1)
    tracker = RLTracker(args.exp_name, args.seed)
    
    model = DreamerV2(args).to(device)
    
    model_params = list(model.encoder.parameters()) + list(model.rssm.parameters()) + \
                   list(model.decoder.parameters()) + list(model.reward_model.parameters())
    optim_model = optim.Adam(model_params, lr=args.learning_rate_model)
    optim_actor = optim.Adam(model.actor_model.parameters(), lr=args.learning_rate_actor)
    optim_value = optim.Adam(model.value_model.parameters(), lr=args.learning_rate_value)
    
    max_steps = 10000 
    obs_b = torch.zeros((max_steps, 4, 64, 64))
    act_b = torch.zeros((max_steps, args.action_dim))
    rew_b = torch.zeros((max_steps, 1))
    
    def process_obs(obs):
        obs = torch.tensor(obs, dtype=torch.float32) / 255.0 
        # Envpool Atari gives [B, 4, 84, 84] by default (stacked grayscale frames)
        return F.interpolate(obs, size=(64, 64))

    print("Pre-filling buffer...")
    obs, _ = env.reset()
    for t in range(500):
        action_idx = np.random.randint(0, args.action_dim, size=(1,))
        action_onehot = F.one_hot(torch.tensor(action_idx), args.action_dim).float()
        
        obs_b[t] = process_obs(obs)[0]
        act_b[t] = action_onehot[0]
        next_obs, reward, term, trunc, _ = env.step(action_idx)
        rew_b[t] = float(reward[0])
        obs = next_obs if not (term[0] or trunc[0]) else env.reset()[0]

    # Main Loop
    global_step = 500
    obs, _ = env.reset()
    h_t = torch.zeros((1, args.deter_size)).to(device)
    s_t = torch.zeros((1, args.category_size * args.class_size)).to(device)
    ep_return, ep_len = 0, 0
    
    for global_step in range(500, args.total_steps):
        # 1. ENVIRONMENT STEP
        with torch.no_grad():
            img_embed = model.encoder(process_obs(obs).to(device))
            post_logits = model.rssm.posterior(h_t, img_embed)
            s_t = st_sample(post_logits).flatten(start_dim=-2)
            
            features = torch.cat([h_t, s_t], dim=-1)
            dist_or_det = model.actor_model(features, deterministic=(global_step < 1000))
            
            if isinstance(dist_or_det, torch.Tensor):
                action_onehot = dist_or_det
            else:
                action_onehot = dist_or_det.sample()
            
            h_t = model.rssm.step_forward(h_t, s_t, action_onehot)

        action_idx = action_onehot.argmax(dim=-1).cpu().numpy()
        next_obs, reward, term, trunc, _ = env.step(action_idx)
        
        idx = global_step % max_steps
        obs_b[idx] = process_obs(obs)[0]
        act_b[idx] = action_onehot[0].cpu()
        rew_b[idx] = float(reward[0])
        
        obs = next_obs
        ep_return += reward[0]
        ep_len += 1
        
        if term[0] or trunc[0]:
            print(f"Step {global_step} | Return: {ep_return:.2f}")
            tracker.log_episode(ep_return, ep_len)
            obs, _ = env.reset()
            h_t = torch.zeros((1, args.deter_size)).to(device)
            s_t = torch.zeros((1, args.category_size * args.class_size)).to(device)
            ep_return, ep_len = 0, 0
            
        tracker.step(1)
        
        # 2. TRAIN STEP
        if global_step % 50 == 0:
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
            s = torch.zeros(args.batch_size, args.category_size * args.class_size).to(device)
            
            kl_loss, recon_loss, reward_loss = 0, 0, 0
            post_states, post_hiddens = [], []
            
            prev_act = torch.zeros(args.batch_size, args.action_dim).to(device)
            
            for t in range(args.seq_len):
                # Step forward uses PREVIOUS action to create context for CURRENT obs
                h = model.rssm.step_forward(h, s, prev_act)
                embed = model.encoder(batch_obs[:, t])
                post_logits = model.rssm.posterior(h, embed)
                s = st_sample(post_logits).flatten(start_dim=-2)
                
                prior_logits = model.rssm.prior(h)
                
                features = torch.cat([h, s], dim=-1)
                kl_loss += calculate_kl_loss(post_logits, prior_logits, args.kl_balance, args.free_nats)
                recon_loss += F.mse_loss(model.decoder(features), batch_obs[:, t])
                reward_loss += F.mse_loss(model.reward_model(features), batch_rew[:, t])
                
                post_states.append(s.detach())
                post_hiddens.append(h.detach())
                prev_act = batch_act[:, t]
                
            model_loss = (0.1 * kl_loss + recon_loss + reward_loss) / args.seq_len
            optim_model.zero_grad()
            model_loss.backward()
            nn.utils.clip_grad_norm_(model_params, 100.0)
            optim_model.step()
            
            # ================== TRAIN ACTOR & CRITIC ==================
            h_imag = torch.stack(post_hiddens).reshape(-1, args.deter_size)
            s_imag = torch.stack(post_states).reshape(-1, args.category_size * args.class_size)
            
            imag_states, imag_hiddens, imag_rewards = [], [], []
            imag_log_probs, imag_entropies = [], []
            
            # Rollout in Imagination
            for t in range(args.horizon):
                features = torch.cat([h_imag, s_imag], dim=-1)
                
                # REINFORCE for discrete actions
                action_dist = model.actor_model(features)
                action = action_dist.sample()
                # Categorical one-hot gradients don't flow directly in vanilla DreamerV2 Actor
                # We use REINFORCE with baseline
                action_ste = action + action_dist.probs - action_dist.probs.detach() # To keep RSSM flowing safely if needed, but actor only needs log prob
                
                h_imag = model.rssm.step_forward(h_imag, s_imag, action_ste)
                prior_logits = model.rssm.prior(h_imag)
                s_imag = st_sample(prior_logits).flatten(start_dim=-2)
                
                feat_imag = torch.cat([h_imag, s_imag], dim=-1)
                imag_rewards.append(model.reward_model(feat_imag).squeeze(-1))
                imag_states.append(s_imag)
                imag_hiddens.append(h_imag)
                imag_log_probs.append(action_dist.log_prob(action))
                imag_entropies.append(action_dist.entropy())
                
            imag_rewards = torch.stack(imag_rewards) 
            imag_features = torch.cat([torch.stack(imag_hiddens), torch.stack(imag_states)], dim=-1)
            imag_values = model.value_model(imag_features).squeeze(-1) 
            imag_log_probs = torch.stack(imag_log_probs)
            imag_entropies = torch.stack(imag_entropies)
            
            returns = compute_lambda_returns(imag_rewards, imag_values, args.gamma, args.lambda_)
            
            # REINFORCE Loss with baseline and entropy bonus
            baseline = imag_values.detach()
            advantage = returns.detach() - baseline
            actor_loss = -(imag_log_probs * advantage).mean() - 1e-4 * imag_entropies.mean()
            
            optim_actor.zero_grad()
            actor_loss.backward(retain_graph=True) 
            nn.utils.clip_grad_norm_(model.actor_model.parameters(), 100.0)
            optim_actor.step()
            
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