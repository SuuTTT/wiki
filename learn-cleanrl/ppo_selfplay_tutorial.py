# ==============================================================================
# SELF-PLAY PPO TUTORIAL (Refer to wiki/SelfPlay-MA.md for theory)
# ==============================================================================

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import tqdm
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import importlib
import supersuit as ss

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    # This environment is multi-agent! 2 Paddles!
    env_id: str = "pong_v3"
    
    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 16  # 16 Total Environments (giving us 32 Agents on field!)
    num_steps: int = 128
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ------------------------------------------------------------------------------
# THE PARAMETER SHARING NETWORK (Self-Play Brain)
# ------------------------------------------------------------------------------
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Since we use FrameStack=4, the input is 4 Grayscale Images. Let's make it robust!
        # SuperSuit reduces color, but let's assume it outputs 4 channels cleanly.
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
        hidden = self.network(x / 255.0)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ------------------------------------------------------------------------------
    # SUPERSUIT FLATTENING (Multi-Agent -> Single-Agent Array)
    # ------------------------------------------------------------------------------
    
    # This brings in PettingZoo's Atari Pong environment (2 agents dynamically tracked)
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    
    # Standard Atari Processing applied mathematically to BOTH agents symmetrically
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B") # Grayscale
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.expand_dims_v0(env) # Adds channel dimension if needed
    env = ss.agent_indicator_v0(env, type_only=False) # Adds visual indicators to help the agent know which paddle it is controlling
    
    # THE FLATTENING TRICK:
    # 16 games * 2 paddles = 32 completely independent "environments" mathematically
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # We ask SuperSuit to literally lie to Gym, telling it we just have 32 standard games
    envs = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gym")
    
    # Because SuperSuit hacked the `gym` space, we manually assign these attributes 
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)

    # 32 Active Agents feeding into ONE Network Brain!
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # NOTE: The actual batch size running through our `get_action_and_value` 
    # will literally be 32 distinct state images!
    # args.num_envs is the physical parallel games instances. PettingZoo gives 2 agents per game.
    total_physical_agents = args.num_envs * 2 
    
    obs = torch.zeros((args.num_steps, total_physical_agents) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, total_physical_agents) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, total_physical_agents)).to(device)
    rewards = torch.zeros((args.num_steps, total_physical_agents)).to(device)
    dones = torch.zeros((args.num_steps, total_physical_agents)).to(device)
    values = torch.zeros((args.num_steps, total_physical_agents)).to(device)

    # Standard PPO Rollout math
    batch_size = int(total_physical_agents * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // batch_size

    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    
    # Because SuperSuit ensures it outputs channel last sometimes (H,W,C), we permute to (C,H,W)
    if len(next_obs.shape) == 4 and next_obs.shape[-1] in [1,3,4]:
        next_obs = next_obs.permute((0, 3, 1, 2))
        
    next_done = torch.zeros(total_physical_agents).to(device)

    # ... The rest of the PPO Training loop is identically mathematically! ...
    # The PPO network has no idea it is optimizing a multi-agent game. 
    # It just optimizes actions based on the images.
