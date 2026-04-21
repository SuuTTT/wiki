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
    
    run_name = f"flat_{args.env}_{int(time.time())}"
    writer = SummaryWriter(f"logs/continuous_study/{run_name}")
    
    agent = ContinuousAC(state_dim, action_dim)
    opt = optim.Adam(agent.parameters(), 3e-4)
    
    print(f"Starting Flat PPO on {args.env} | Log: {run_name}")
    
    for ep in range(args.episodes):
        s, _ = env.reset(); done = False; total_rew = 0
        hist = []
        
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            mu_a, sig_a, val = agent(s_t)
            dist = Normal(mu_a, sig_a)
            action = dist.sample()
            
            act_clipped = torch.clamp(action, env.action_space.low[0], env.action_space.high[0]).detach().numpy()
            
            ns, r, term, trunc, _ = env.step(act_clipped)
            done = term or trunc
            
            hist.append((val, dist.log_prob(action), r))
            s = ns; total_rew += r

        if len(hist) > 0:
            vals, log_probs, rews = zip(*hist)
            vals = torch.stack(vals).view(-1)
            R = []; curr_r = 0
            for r_val in reversed(rews):
                curr_r = r_val + 0.99 * curr_r
                R.insert(0, curr_r)
            R = torch.tensor(R, dtype=torch.float32)
            adv = R - vals.detach()
            log_p_sum = torch.stack(log_probs).sum(dim=-1)
            loss = -(log_p_sum * adv).mean() + 0.5 * F.mse_loss(vals, R)
            opt.zero_grad(); loss.backward(); opt.step()

        if ep % 10 == 0:
            print(f"[{args.env}-Flat] Ep {ep:4d} | Reward: {total_rew:8.2f}")
            writer.add_scalar("charts/episodic_return", total_rew, ep)

    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    train(args)
