import argparse
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import random
import time
import importlib

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class Args:
    env_id: str = "pistonball_v6"
    seed: int = 1
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

args = Args()
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
num_updates = args.total_timesteps // args.batch_size

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.clone().float()
        if x.shape[-1] == 4:
            x[:, :, :, [0, 1, 2, 3]] /= 255.0
            x = x.permute((0, 3, 1, 2))
        else:
            x /= 255.0
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        x = x.clone().float()
        if x.shape[-1] == 4:
            x[:, :, :, [0, 1, 2, 3]] /= 255.0
            x = x.permute((0, 3, 1, 2))
        else:
            x /= 255.0
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.env_id.startswith("pong"):
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
else:
    # Use Pistonball as the fallback because it relies on zero external C++ dependencies!
    env = importlib.import_module(f"pettingzoo.butterfly.{args.env_id}").parallel_env(continuous=False)

env = ss.color_reduction_v0(env, mode="B") # Grayscale
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.pettingzoo_env_to_vec_env_v1(env)
envs = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gymnasium")

envs.single_observation_space = envs.observation_space
envs.single_action_space = envs.action_space
envs.is_vector_env = True

# Native Gym Return Wrapper substitute for custom wrappers issue
# Pistonball has 20 agents originally, Supersuit flattens
TOTAL_PARALLEL_AGENTS = envs.num_envs
episodic_returns = np.zeros(TOTAL_PARALLEL_AGENTS, dtype=np.float32)

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

obs = torch.zeros((args.num_steps, TOTAL_PARALLEL_AGENTS) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, TOTAL_PARALLEL_AGENTS) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, TOTAL_PARALLEL_AGENTS)).to(device)
rewards = torch.zeros((args.num_steps, TOTAL_PARALLEL_AGENTS)).to(device)
dones = torch.zeros((args.num_steps, TOTAL_PARALLEL_AGENTS)).to(device)
values = torch.zeros((args.num_steps, TOTAL_PARALLEL_AGENTS)).to(device)

global_step = 0
start_time = time.time()
next_obs_numpy, _ = envs.reset()
next_obs = torch.tensor(next_obs_numpy).to(device)
next_done = torch.zeros(TOTAL_PARALLEL_AGENTS).to(device)

for update in range(1, num_updates + 1):
    frac = 1.0 - (update - 1.0) / num_updates
    optimizer.param_groups[0]["lr"] = frac * args.learning_rate

    for step in range(0, args.num_steps):
        global_step += TOTAL_PARALLEL_AGENTS
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # ENV STEP (Gymnasium tuple format via super-suit)
        next_obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
        done = np.logical_or(term, trunc)
        
        episodic_returns += reward
        for i, d in enumerate(done):
            if d:
                # Log when an agent completes a multi-agent episode!
                if update % 5 == 0 and i == 0:
                    print(f"global_step={global_step}, selfplay_ep_return={episodic_returns[i]}")
                episodic_returns[i] = 0

        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.tensor(next_obs_np).to(device), torch.tensor(done, dtype=torch.float32).to(device)

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    b_inds = np.arange(args.batch_size)
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    if update % 25 == 0:
        print(f"Iteration {update}/{num_updates}, SPS: {int(global_step / (time.time() - start_time))}")

print("✅ Multi-Agent Self-Play Tutorial Completed!")
