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
    
    # PPO parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
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
        
        # Encoder: (Batch, 3, 64, 64) -> (Batch, 256, 4, 4) -> (Batch, 4096)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # Mu and logvar for the reparameterization trick
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim) 
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder: (Batch, latent_dim) -> (Batch, 3, 64, 64)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)), 
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
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

class ActorCritic(nn.Module):
    """
    Replaces the linear Controller with a robust PPO Actor-Critic MLP utilizing the
    concatenated latent state `z_t` and RNN hidden state `h_t`.
    """
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
        super().__init__()
        # Flattened state input: (z_t, h_t)
        in_features = latent_dim + hidden_dim
        
        self.actor = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        # We don't forward directly, primarily split via get_value and get_action
        return x

    def get_value(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return self.critic(x)

    def get_action_and_value(self, z, h, action=None):
        x = torch.cat([z, h], dim=-1)
        
        # Actor
        hidden_actor = self.actor(x)
        action_mean = self.actor_mean(hidden_actor)
        # Binds continuous outputs (CarRacing expects continuous [-1,1] or similar)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        # Critic
        value = self.critic(x)
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Initializing Fast MLP-PPO Controller over World Models features on {args.env_id}")
    
    import envpool
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../learn-cleanrl")))
    from cleanrl_utils.logger import RLTracker
    
    # Fast parallel environment execution
    num_envs = 16
    envs = envpool.make(args.env_id, env_type="gymnasium", num_envs=num_envs)
    
    # 0. Initialize Logger using README standards
    tracker = RLTracker(exp_name=args.exp_name, seed=args.seed)
    
    # 1. Initialize Combined Architecture Components
    vae = VAE(args.latent_dim).to(device)
    rnn = MDNRNN(args.latent_dim, action_dim=3, hidden_dim=args.rnn_hidden_dim, num_mix=args.num_gaussian_mix).to(device)
    agent = ActorCritic(args.latent_dim, args.rnn_hidden_dim, action_dim=3).to(device)
    
    # Optimizers
    combined_optimizer = optim.Adam(
        list(vae.parameters()) + list(rnn.parameters()) + list(agent.parameters()), 
        lr=3e-4, eps=1e-5
    )
    
    print("\n--- Next Gen Controller Enhancement ---")
    print(f"[{time.strftime('%H:%M:%S')}] Swapped linear CMA-ES Controller to a direct MLP PPO algorithm.")
    print(f"[{time.strftime('%H:%M:%S')}] Spawned {num_envs} identical instances of {args.env_id} via EnvPool in C++.")
    
    # --- MAIN TRAINING LOOP ---
    print(f"[{time.strftime('%H:%M:%S')}] Launching Full Training Pipeline...")
    obs, info = envs.reset()
    
    # Simple image preprocessing (numpy to torch, permute, resize)
    def preprocess_obs(o):
        # Envpool CarRacing-v2 obs are (B, 96, 96, 3) uint8
        o = torch.tensor(o, dtype=torch.float32).to(device) / 255.0
        o = o.permute(0, 3, 1, 2)  # (B, 3, 96, 96)
        o = F.interpolate(o, size=(64, 64)) # (B, 3, 64, 64)
        return o

    total_steps = 1_000_000
    num_steps_per_rollout = 128
    hidden = (torch.zeros(1, num_envs, args.rnn_hidden_dim).to(device),
              torch.zeros(1, num_envs, args.rnn_hidden_dim).to(device))
              
    global_step = 0
    start_time = time.time()
    
    # Manual episodic return tracking
    current_episode_return = np.zeros(num_envs, dtype=np.float32)
    current_episode_length = np.zeros(num_envs, dtype=np.int32)
    
    import tqdm
    
    # PPO Rollout Storage
    num_updates = total_steps // (num_envs * num_steps_per_rollout)
    
    # Pre-allocate rollout tensors
    obs_buf = torch.zeros((num_steps_per_rollout, num_envs, 3, 64, 64)).to(device)
    actions_buf = torch.zeros((num_steps_per_rollout, num_envs, 3)).to(device)
    logprobs_buf = torch.zeros((num_steps_per_rollout, num_envs)).to(device)
    rewards_buf = torch.zeros((num_steps_per_rollout, num_envs)).to(device)
    dones_buf = torch.zeros((num_steps_per_rollout, num_envs)).to(device)
    values_buf = torch.zeros((num_steps_per_rollout, num_envs)).to(device)
    
    # Need to store the RNN hiddens per step for batched Actor updates:
    hiddens_buf = torch.zeros((num_steps_per_rollout, num_envs, args.rnn_hidden_dim)).to(device)
    
    for update in tqdm.tqdm(range(1, num_updates + 1), desc="Training updates"):
        
        for step in range(num_steps_per_rollout):
            proc_obs = preprocess_obs(obs)
            
            with torch.no_grad():
                _, mu, logvar = vae(proc_obs)
                z = vae.reparameterize(mu, logvar)
                
                # Fetch actions
                action, logprob, _, value = agent.get_action_and_value(z, hidden[0].squeeze(0))
                
                # Store rollouts
                obs_buf[step] = proc_obs
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value.flatten()
                hiddens_buf[step] = hidden[0].squeeze(0)
                
                # Step RNN imagination forward
                _, _, _, next_hidden = rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
                hidden = next_hidden

            # Take step
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            obs = next_obs
            
            # Manual episode tracking
            current_episode_return += reward
            current_episode_length += 1
            
            dones_buf[step] = torch.tensor(terminated | truncated, dtype=torch.float32).to(device)
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            
            global_step += num_envs
            tracker.step(num_envs)
            
            finished_envs = terminated | truncated
            if finished_envs.any():
                for idx in np.where(finished_envs)[0]:
                    tracker.log_episode(current_episode_return[idx], current_episode_length[idx])
                    if (update * step) % 100 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] Update {update} - Episodic Return: {current_episode_return[idx]:.2f}")
                    # Reset the manual tracker
                    current_episode_return[idx] = 0.0
                    current_episode_length[idx] = 0

        # --- GAE (Generalized Advantage Estimation) ---
        with torch.no_grad():
            proc_obs = preprocess_obs(obs)
            _, mu_next, logvar_next = vae(proc_obs)
            z_next = vae.reparameterize(mu_next, logvar_next)
            next_value = agent.get_value(z_next, hidden[0].squeeze(0)).flatten()
            
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps_per_rollout)):
                if t == num_steps_per_rollout - 1:
                    nextnonterminal = 1.0 - dones_buf[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t+1]
                    nextvalues = values_buf[t+1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # --- JOINT OPTIMIZATION (PPO + VAE) epochs ---
        b_obs = obs_buf.reshape((-1, 3, 64, 64))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1, 3))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_hiddens = hiddens_buf.reshape((-1, args.rnn_hidden_dim))

        # Flatten batch sizes for iteration
        b_inds = np.arange(num_envs * num_steps_per_rollout)
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # In a full run, we would chunk this into mini-batches. For clean 16x128 (2048), doing full batch is okay for CartRacing scaling.
            for start in range(0, len(b_inds), 1024):
                end = start + 1024
                mb_inds = b_inds[start:end]
                
                # Re-compute latent distributions
                reconstructed, mu, logvar = vae(b_obs[mb_inds])
                z = vae.reparameterize(mu, logvar)
                
                # Fetch new action logprobs and values from updated graph
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(z, b_hiddens[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if mb_advantages.std() > 1e-8: # Simple Advantage Normalization
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PPO Clipped Critic Loss
                v_loss_unclipped = (newvalue.flatten() - b_returns[mb_inds]) ** 2
                v_clipped = b_returns[mb_inds] + torch.clamp(
                    newvalue.flatten() - b_returns[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # PPO Actor Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                ent_loss = entropy.mean()
                agent_loss = pg_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef
                
                # VAE Loss (Reconstruction + KL)
                recon_loss = F.mse_loss(reconstructed, b_obs[mb_inds])
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mb_inds.shape[0]
                vae_loss = recon_loss + 0.0001 * kld_loss

                # Optimize joint architecture
                combined_optimizer.zero_grad()
                total_loss = vae_loss + agent_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()) + list(vae.parameters()), args.max_grad_norm)
                combined_optimizer.step()

        # Log metrics at the end of the update block
        tracker.log_metrics("losses", {
            "vae_recon_loss": recon_loss.item(),
            "vae_kld": kld_loss.item(),
            "agent_value_loss": v_loss.item(),
            "agent_policy_loss": pg_loss.item(),
            "agent_entropy": ent_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "agent_value": newvalue.mean().item()
        })
        tracker.log_metrics("debug", {
            "latent_z_mean_abs": z.abs().mean().item(),
            "latent_z_std": z.std().item(),
            "rnn_hidden_mean_abs": b_hiddens.abs().mean().item(),
        })
        tracker.log_sps()

    tracker.save_checkpoint({
        'vae': vae.state_dict(),
        'rnn': rnn.state_dict(),
        'agent': agent.state_dict()
    })
    tracker.close()

