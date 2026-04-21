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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_in, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_in, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_out), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_out))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def train(args):
    def make_env(env_id):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.NormalizeReward(env)
            return env
        return thunk

    env = make_env(args.env)()
    run_name = f"ppo_final_stable_{args.env}_{int(time.time())}"
    writer = SummaryWriter(f"logs/continuous_study/{run_name}")
    
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    print(f"Starting FINAL Stable PPO on {args.env} | Device: {device}")
    global_step = 0
    start_time = time.time()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episodic_reward = 0
        b_obs, b_actions, b_logprobs, b_rewards, b_values, b_dones = [], [], [], [], [], []
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            next_obs, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
            done = term or trunc
            b_obs.append(obs_t)
            b_actions.append(action)
            b_logprobs.append(logprob)
            b_rewards.append(reward)
            b_values.append(value)
            b_dones.append(done)
            obs = next_obs
            episodic_reward += reward
            global_step += 1

        with torch.no_grad():
            returns = []
            values = torch.stack(b_values).squeeze()
            next_value = 0
            gae = 0
            for i in reversed(range(len(b_rewards))):
                delta = b_rewards[i] + args.gamma * next_value * (1 - b_dones[i]) - values[i]
                gae = delta + args.gamma * args.gae_lambda * (1 - b_dones[i]) * gae
                returns.insert(0, gae + values[i])
                next_value = values[i]
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            advantages = returns - values

        s_obs = torch.cat(b_obs)
        s_actions = torch.cat(b_actions)
        s_logprobs = torch.cat(b_logprobs)
        s_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(args.update_epochs):
            _, new_logprob, entropy, new_value = agent.get_action_and_value(s_obs, s_actions)
            ratio = torch.exp(new_logprob - s_logprobs)
            surr1 = ratio * s_advantages
            surr2 = torch.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * s_advantages
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.mse_loss(new_value.squeeze(), returns)
            ent_loss = entropy.mean()
            loss = pi_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if ep % 20 == 0:
            print(f"[{args.env}] Ep {ep:4d} | Rew: {episodic_reward:.2f}")
            writer.add_scalar("charts/episodic_return", episodic_reward, global_step)

    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad_norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=10)
    args = parser.parse_args()
    train(args)
