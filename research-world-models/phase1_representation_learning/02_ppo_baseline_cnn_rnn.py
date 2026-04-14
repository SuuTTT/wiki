import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import envpool
import tqdm
import sys
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../learn-cleanrl")))
from cleanrl_utils.logger import RLTracker

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True
    env_id: str = "CarRacing-v2"
    
    # PPO Parameters
    total_steps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rnn_hidden_dim: int = 256

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        # Matches the exact convolutional capacity of our VAE encoder minus the variance sampling
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 4, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, 4, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(256 * 4 * 4, 256, batch_first=True)
        
        self.actor_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_states(self, x, hidden):
        hidden_network = self.network(x)
        # Sequence length of 1 for step-by-step unrolling
        hidden_network = hidden_network.unsqueeze(1) 
        lstm_out, hidden = self.lstm(hidden_network, hidden)
        return lstm_out.squeeze(1), hidden

    def get_value(self, x, hidden):
        lstm_out, _ = self.get_states(x, hidden)
        return self.critic(lstm_out)

    def get_action_and_value(self, x, hidden, action=None):
        lstm_out, hidden = self.get_states(x, hidden)
        
        action_mean = self.actor_mean(lstm_out)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(lstm_out), hidden

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    print(f"Initializing Standard End-to-End PPO+RNN Baseline on {args.env_id}")
    
    envs = envpool.make(args.env_id, env_type="gymnasium", num_envs=args.num_envs)
    tracker = RLTracker(exp_name=args.exp_name, seed=args.seed)
    
    agent = Agent(action_dim=3).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    obs, info = envs.reset()
    
    def preprocess_obs(o):
        o = torch.tensor(o, dtype=torch.float32).to(device) / 255.0
        o = o.permute(0, 3, 1, 2)  
        o = F.interpolate(o, size=(64, 64))
        return o

    # Rollout buffer
    obs_buf = torch.zeros((args.num_steps, args.num_envs, 3, 64, 64)).to(device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs, 3)).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    hiddens_buf = torch.zeros((args.num_steps, args.num_envs, args.rnn_hidden_dim)).to(device)
    
    hidden = (torch.zeros(1, args.num_envs, args.rnn_hidden_dim).to(device),
              torch.zeros(1, args.num_envs, args.rnn_hidden_dim).to(device))

    global_step = 0
    current_episode_return = np.zeros(args.num_envs, dtype=np.float32)
    current_episode_length = np.zeros(args.num_envs, dtype=np.int32)
    num_updates = args.total_steps // (args.num_envs * args.num_steps)
    
    for update in tqdm.tqdm(range(1, num_updates + 1)):
        for step in range(args.num_steps):
            proc_obs = preprocess_obs(obs)
            
            with torch.no_grad():
                action, logprob, _, value, next_hidden = agent.get_action_and_value(proc_obs, hidden)
                
                obs_buf[step] = proc_obs
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value.flatten()
                hiddens_buf[step] = hidden[0].squeeze(0) # store explicitly for bptt
                hidden = next_hidden

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            obs = next_obs
            
            current_episode_return += reward
            current_episode_length += 1
            
            dones_buf[step] = torch.tensor(terminated | truncated, dtype=torch.float32).to(device)
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            
            global_step += args.num_envs
            tracker.step(args.num_envs)
            
            finished_envs = terminated | truncated
            if finished_envs.any():
                for idx in np.where(finished_envs)[0]:
                    tracker.log_episode(current_episode_return[idx], current_episode_length[idx])
                    if (update * step) % 100 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] Step {global_step} - PPO Baseline Return: {current_episode_return[idx]:.2f}")
                    current_episode_return[idx] = 0.0
                    current_episode_length[idx] = 0

        # GAE
        with torch.no_grad():
            proc_obs = preprocess_obs(obs)
            next_value = agent.get_value(proc_obs, hidden).flatten()
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - dones_buf[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t+1]
                    nextvalues = values_buf[t+1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flat tensor creation
        b_obs = obs_buf.reshape((-1, 3, 64, 64))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1, 3))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_hiddens = hiddens_buf.reshape((-1, args.rnn_hidden_dim))

        # NOTE: For true LSTM BPTT we shouldn't flatten perfectly into 1D minibatches, 
        # but matching our previous pseudo-world-model structural simplifications here:
        b_inds = np.arange(args.num_envs * args.num_steps)
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), 1024):
                end = start + 1024
                mb_inds = b_inds[start:end]
                
                # Re-pack hidden (1, batch, dim) and process
                mb_hiddens = (b_hiddens[mb_inds].unsqueeze(0), torch.zeros_like(b_hiddens[mb_inds].unsqueeze(0)))
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], mb_hiddens, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if mb_advantages.std() > 1e-8:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                v_loss_unclipped = (newvalue.flatten() - b_returns[mb_inds]) ** 2
                v_clipped = b_returns[mb_inds] + torch.clamp(newvalue.flatten() - b_returns[mb_inds], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        tracker.log_metrics("losses", {
            "agent_value_loss": v_loss.item(),
            "agent_policy_loss": pg_loss.item(),
            "agent_entropy": ent_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs)
        })
        tracker.log_metrics("debug", {
            "rnn_hidden_mean_abs": b_hiddens.abs().mean().item(),
        })
        tracker.log_sps()

    tracker.close()