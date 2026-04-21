import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import time
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.actor = nn.Linear(256, n_out)
        self.critic = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.zeros(1, n_out))

    def get_value(self, x):
        return self.critic(self.fc(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.fc(x)
        mu = self.actor(hidden)
        # Scale mu to action range if needed (LunarLander is -1 to 1, Pendulum is -2 to 2)
        # Note: Pendulum-v1 needs tanh * 2, but for simplicity we'll let it learn the scale
        std = torch.exp(self.log_std)
        probs = Normal(mu, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)

def train(args):
    env = gym.make(args.env)
    # Correct action scaling for Pendulum
    action_scale = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
    
    run_name = f"ppo_baseline_{args.env}_{int(time.time())}"
    writer = SummaryWriter(f"logs/continuous_study/{run_name}")
    
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    print(f"Starting High-Stability PPO on {args.env} | Device: {device}")
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        # Buffers for the episode
        b_obs, b_actions, b_logprobs, b_rewards, b_values, b_dones = [], [], [], [], [], []
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            
            # Scale action for Pendulum specifically if needed, but LunarLander is [-1, 1]
            # Here we let the network learn the output but clip to safety
            act_env = torch.clamp(action, -1.0, 1.0)
            if args.env == "Pendulum-v1":
                act_env = act_env * 2.0
                
            next_obs, reward, term, trunc, _ = env.step(act_env.cpu().numpy().flatten())
            done = term or trunc
            
            b_obs.append(obs_t)
            b_actions.append(action)
            b_logprobs.append(logprob)
            b_rewards.append(reward)
            b_values.append(value)
            b_dones.append(done)
            
            obs = next_obs
            total_reward += reward

        # Batch Update (Standard PPO logic)
        with torch.no_grad():
            next_value = 0 # End of episode
            returns = []
            curr_return = next_value
            for r, d in zip(reversed(b_rewards), reversed(b_dones)):
                curr_return = r + 0.99 * curr_return * (1 - d)
                returns.insert(0, curr_return)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            advantages = returns - torch.stack(b_values).squeeze()

        # Optimize for K epochs
        s_obs = torch.cat(b_obs)
        s_actions = torch.cat(b_actions)
        s_logprobs = torch.cat(b_logprobs)
        s_returns = returns
        s_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(10):
            _, new_logprob, entropy, new_value = agent.get_action_and_value(s_obs, s_actions)
            ratio = torch.exp(new_logprob - s_logprobs)
            
            surr1 = ratio * s_advantages
            surr2 = torch.clip(ratio, 0.8, 1.2) * s_advantages
            pi_loss = -torch.min(surr1, surr2).mean()
            
            v_loss = 0.5 * F.mse_loss(new_value.squeeze(), s_returns)
            ent_loss = entropy.mean()
            
            loss = pi_loss + 0.5 * v_loss - 0.01 * ent_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

        if ep % 20 == 0:
            print(f"[{args.env}] Ep {ep:4d} | Reward: {total_reward:8.2f}")
            writer.add_scalar("charts/episodic_return", total_reward, ep)
            writer.add_scalar("losses/policy_loss", pi_loss.item(), ep)
            writer.add_scalar("losses/value_loss", v_loss.item(), ep)
            writer.add_scalar("losses/entropy", ent_loss.item(), ep)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], ep)
            writer.add_scalar("charts/std", torch.exp(agent.log_std).mean().item(), ep)

    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=1500)
    args = parser.parse_args()
    train(args)
