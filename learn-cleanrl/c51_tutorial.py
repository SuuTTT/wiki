# ==============================================================================
# C51 TUTORIAL (Refer to wiki/C51.md for theory)
# ==============================================================================

import os
import random
import sys
import time
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cleanrl')))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import tqdm
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = True  # We enable video rendering!

    # Algorithm specific arguments
    # THEORY: We return to CartPole-v1 for Discrete Action Spaces.
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    
    # THEORY: The Atoms (wiki/C51.md - Section 2)
    n_atoms: int = 101           # The number of bins the return distribution is chopped into.
    v_min: float = -100          # The absolute theoretical minimum return possible in the environment.
    v_max: float = 100           # The absolute theoretical maximum return possible in the environment.
    
    buffer_size: int = 10000
    gamma: float = 0.99
    
    # We revert to Hard Target Network Updates (like DQN) for discrete C51, rather than Soft updates (tau).
    target_network_frequency: int = 500
    batch_size: int = 128
    
    # Epsilon-Greedy Exploration
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    
    learning_starts: int = 10000
    train_frequency: int = 10


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", lambda x: x % 50 == 0)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


# ------------------------------------------------------------------------------
# Neural Network Architecture
# ------------------------------------------------------------------------------

# THEORY: The Distributional Network (wiki/C51.md - Section 2)
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        
        # We explicitly calculate the exact values for all 101 bins.
        # e.g., if v_min=-10, v_max=10, n_atoms=3 -> [-10, 0, 10]
        # This creates a non-trainable tensor that rides along with the Network.
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        
        self.n = env.single_action_space.n
        
        # In standard DQN, we output `self.n` Q-values (one for each action).
        # In C51, we output `101 probabilities` for EVERY SINGLE ACTION.
        # Output size: num_actions * 101
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms), # <--- Massive structural difference!
        )

    def get_action(self, x, action=None):
        """Pass state forward and return the predicted Probabilities AND the chosen Action."""
        logits = self.network(x)
        
        # Reshape the massive flat output into [BatchSize, Num_Actions, 101_Atoms]
        # Then, apply Softmax across the Atoms dimension so they mathematically sum perfectly to 1.0.
        # This converts raw Neural Network floats into a Probability Mass Function (PMF).
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        
        # THEORY: How do we actually pick an Action if we only have Probabilities?
        # We calculate the Expected Value! Multiply every probability by its bin's actual value, and sum them up.
        # Expected Value = (0.01 * -100) + (0.5 * 0) + (0.49 * 100) = 48.0 Q-Value.
        q_values = (pmfs * self.atoms).sum(2)
        
        if action is None:
            # We pick the action with the highest Expected Value, just like DQN.
            action = torch.argmax(q_values, 1)
            
        # We return the Action, and the exact 101-bin Probability Distribution for that specific Action.
        return action, pmfs[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    writer = SummaryWriter(f"runs/{run_name}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    # Initialize Distributional Q-Networks
    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    # Adam optimizer with a tiny epsilon scaling trick common in C51
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # THE TRAINING LOOP
    obs, _ = envs.reset(seed=args.seed)
    
    progress_bar = tqdm.tqdm(range(args.total_timesteps), desc="C51 Training", leave=True)
    for global_step in progress_bar:
        
        # 1. EXPLORATION VS EXPLOITATION (Exactly like DQN)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # 2. ENVIRONMENT STEP
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    progress_bar.set_postfix(epsilon=f"{epsilon:.2f}", episodic_return=info['episode']['r'][0] if isinstance(info['episode']['r'], np.ndarray) else info['episode']['r'])
                    tqdm.tqdm.write(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        # Save data to reply buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # ==============================================================================
        # DISTRIBUTIONAL TRAINING (wiki/C51.md - Section 3)
        # ==============================================================================
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            
            with torch.no_grad():
                # Get the 101 probabilities of the NEXT state from the Target Network.
                _, next_pmfs = target_network.get_action(data.next_observations)
                
                # Shift the ATOMS themselves! Instead of bins representing [-100, ..., 100],
                # If reward=10 and gamma=0.99, the target bins shift to [-89, ..., +109]
                next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                
                # THEORY: The Projection Step (wiki/C51.md - Section 3)
                delta_z = target_network.atoms[1] - target_network.atoms[0]
                # Clamp the shifted bins so they don't explode past v_min and v_max.
                tz = next_atoms.clamp(args.v_min, args.v_max)

                # Figure out exactly how far the shifted bins now sit BETWEEN our 101 fixed bins mathematically.
                b = (tz - args.v_min) / delta_z
                l = b.floor().clamp(0, args.n_atoms - 1) # The Lower bin index
                u = b.ceil().clamp(0, args.n_atoms - 1)  # The Upper bin index
                
                # Distribute the probabilities proportionally to the nearest neighboring bins!
                d_m_l = (u + (l == u).float() - b) * next_pmfs
                d_m_u = (b - l) * next_pmfs
                
                # Initialize an empty probability distribution and cleanly inject the split probabilities in place.
                target_pmfs = torch.zeros_like(next_pmfs)
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

            # Get the Current Primary Network's 101 predictions for the Current State & Real Action took.
            _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
            
            # THEORY: Cross-Entropy Loss (wiki/C51.md - Section 4)
            # Log the predicted probability, clamp it slightly to prevent ln(0)=Infinity explosions.
            # Multiply by Target actual probability, sum across all 101 bins, and average across batch.
            loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

            # Optimize the network!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                writer.add_scalar("losses/loss", loss.item(), global_step)
                old_val = (old_pmfs * q_network.atoms).sum(1)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

        # 4. HARD TARGET UPDATE (Just like DQN)
        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()