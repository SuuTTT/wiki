# ==============================================================================
# RAINBOW DQN TUTORIAL (Refer to wiki/Rainbow.md for theory)
# ==============================================================================

import collections
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import tqdm
from cleanrl_utils.logger import RLTracker
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False  # We want video!

    # Algorithm specific arguments
    env_id: str = "CartPole-v1" # Stick to basics
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 100000
    gamma: float = 0.99
    
    # 1. Double DQN - Separate Target network, updated every N steps.
    target_network_frequency: int = 500
    batch_size: int = 128
    learning_starts: int = 10000
    train_frequency: int = 10
    
    # 2. Multi-Step Learning - Bootstrapping forward 3 steps instead of 1.
    n_step: int = 3
    
    # 3. Prioritized Experience Replay (PER)
    prioritized_replay_alpha: float = 0.5   # How much prioritization is used (0 = uniform, 1 = total priority)
    prioritized_replay_beta: float = 0.4    # Importance sampling weight, transitions from 0.4 to 1.0
    prioritized_replay_eps: float = 1e-6    # Tiny offset so priority is never literally zero
    
    # 4. Distributional RL (C51)
    n_atoms: int = 51
    v_min: float = -10
    v_max: float = 10

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
# THE SIX EXTENSIONS OF RAINBOW
# ------------------------------------------------------------------------------

# --- 1. Noisy Networks (Replacing Epsilon-Greedy) ---
class NoisyLinear(nn.Module):
    """
    Parametric Noise! Instead of manually writing `if random() < epsilon`, we inject
    trainable noise into the Linear layer. The network learns *how much* it wants to explore.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Normal weights & biases
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # We manually refresh the epsilon Gaussian noise every single backward pass.
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        # If in training mode, add the Noisy parameters to the primary parameters
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# --- 2. Dueling Networks & 3. Distributional RL (C51) ---
class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = env.single_action_space.n
        # 3. Distributional: Define 51 Fixed Atoms/Bins for Probability distribution mapping
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        # Base Feature Extractor
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )

        # 2. Dueling Architecture: Split into Value Stream and Advantage Stream
        # Both utilize the custom NoisyLinear layer for exploration!
        self.value_head = nn.Sequential(
            NoisyLinear(84, 128), nn.ReLU(), NoisyLinear(128, n_atoms)
        )
        self.advantage_head = nn.Sequential(
            NoisyLinear(84, 128), nn.ReLU(), NoisyLinear(128, n_atoms * self.n_actions)
        )

    def forward(self, x):
        h = self.network(x)
        # Value answers: How good is the state naturally?
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        # Advantage answers: Which actions are specifically better than others right now?
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        
        # Recombine them: Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        # Convert raw values across 51 bins to proper Probabilities (PMF)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear): layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear): layer.reset_noise()

# --- Segment Trees (Required for PER fast lookups) ---
class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def total(self):
        return self.tree[0]

    def retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < len(self.tree):
            left = idx * 2 + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)

class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.full(2 * capacity - 1, float("inf"), dtype=np.float32)

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])
            parent = (parent - 1) // 2

    def min(self):
        return self.tree[0]

PrioritizedBatch = collections.namedtuple(
    "PrioritizedBatch", ["observations", "actions", "rewards", "next_observations", "dones", "indices", "weights"]
)

# --- 4. Prioritized Experience Replay (PER) & 5. Multi-Step Returns ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, device, n_step, gamma, alpha, beta, eps):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # Modified for Cartpole Floats instead of Atari uint8 images
        self.buffer_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.buffer_next_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.buffer_actions = np.zeros(capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer_dones = np.zeros(capacity, dtype=np.bool_)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0 # New items get max priority ensuring they are sampled early on.

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        self.n_step_buffer = deque(maxlen=n_step) # Bootstrapping buffer

    def _get_n_step_info(self):
        # Accumulate discounted rewards across the entire n_step queue.
        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for i in range(len(self.n_step_buffer)):
            reward += self.gamma**i * self.n_step_buffer[i][2]
            if self.n_step_buffer[i][4]:
                next_obs = self.n_step_buffer[i][3]
                done = True
                break
        return reward, next_obs, done

    def add(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_next_obs[idx] = next_obs
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_dones[idx] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        indices = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        samples = {
            "observations": torch.from_numpy(self.buffer_obs[indices]).to(self.device),
            "actions": torch.from_numpy(self.buffer_actions[indices]).to(self.device).unsqueeze(1),
            "rewards": torch.from_numpy(self.buffer_rewards[indices]).to(self.device).unsqueeze(1),
            "next_observations": torch.from_numpy(self.buffer_next_obs[indices]).to(self.device),
            "dones": torch.from_numpy(self.buffer_dones[indices]).to(self.device).unsqueeze(1),
        }

        # Calculate importance sampling weights
        probs = np.array([self.sum_tree.tree[idx + self.capacity - 1] for idx in indices])
        weights = (self.size * probs / p_total) ** -self.beta
        weights = weights / weights.max()  # Normalize to 1.0 max to prevent massive gradient scaling
        
        samples["weights"] = torch.from_numpy(weights).to(self.device).unsqueeze(1)
        samples["indices"] = indices

        return PrioritizedBatch(**samples)

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())
        for idx, priority in zip(indices, priorities):
            priority = priority**self.alpha
            self.sum_tree.update(idx, priority)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    tracker = RLTracker(args.exp_name, args.seed)
    writer = tracker.writer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    q_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    
    target_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
        args.n_step,
        args.gamma,
        args.prioritized_replay_alpha,
        args.prioritized_replay_beta,
        args.prioritized_replay_eps,
    )

    obs, _ = envs.reset(seed=args.seed)
    progress_bar = tqdm.tqdm(range(args.total_timesteps), desc="Rainbow Training")
    
    for global_step in progress_bar:
        
        # Importance sampling beta increases from 0.4 to 1.0 slowly.
        rb.beta = linear_schedule(args.prioritized_replay_beta, 1.0, args.total_timesteps, global_step)

        # 6. ACTION SELECTION: No epsilon-greedy! We just do an argmax directly. 
        # The NoisyLinear layers provide random exploration inherently.
        with torch.no_grad():
            q_network.reset_noise() # Resample the weight injection noise
            
            # Output is PMF probabilities. Multiply by fixed atoms, then sum to get Expected Value.
            pmfs = q_network(torch.Tensor(obs).to(device))
            q_values = (pmfs * q_network.support).sum(2)
            actions = torch.argmax(q_values, 1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "_episode" in infos:
                for idx, d in enumerate(infos["_episode"]):
                    if d:
                        r = infos["episode"]["r"][idx].item() if hasattr(infos["episode"]["r"][idx], "item") else infos["episode"]["r"][idx]
                        l = infos["episode"]["l"][idx].item() if hasattr(infos["episode"]["l"][idx], "item") else infos["episode"]["l"][idx]
                        tracker.log_episode(r, l)
                        if 'progress_bar' in locals():
                            progress_bar.set_postfix(episodic_return=f"{r:.2f}")


        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
                
        rb.add(obs[0], actions[0], rewards[0], real_next_obs[0], terminations[0])
        obs = next_obs

        # ==============================================================================
        # TRAINING LOOP
        # ==============================================================================
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            
            with torch.no_grad():
                target_network.reset_noise()
                q_network.reset_noise()
                
                # --- Double DQN Component ---
                # 1. Use the Primary Network to select the BEST next action.
                next_pmfs_q = q_network(data.next_observations)
                next_q_values = (next_pmfs_q * q_network.support).sum(2)
                next_actions = torch.argmax(next_q_values, dim=1)
                
                # 2. Use the Target Network to evaluate that action!
                next_pmfs_target = target_network(data.next_observations)
                # Next PMF is the Probability distribution for just the chosen action
                next_pmfs = next_pmfs_target[torch.arange(args.batch_size), next_actions]

                # --- Multi-Step C51 Component ---
                next_atoms = data.rewards + (args.gamma ** args.n_step) * target_network.support * (1 - data.dones.float())
                tz = next_atoms.clamp(args.v_min, args.v_max)

                # Projection step as studied in C51
                delta_z = q_network.delta_z
                b = (tz - args.v_min) / delta_z
                l = b.floor().clamp(0, args.n_atoms - 1)
                u = b.ceil().clamp(0, args.n_atoms - 1)

                d_m_l = (u.float() + (l == b).float() - b) * next_pmfs
                d_m_u = (b - l) * next_pmfs

                target_pmfs = torch.zeros_like(next_pmfs)
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

            # Get current probabilities from primary network for the action we ACTUALLY took.
            dist = q_network(data.observations)
            old_pmfs = dist[torch.arange(args.batch_size), data.actions.flatten()]
            
            # --- Prioritized Experience Replay Component ---
            # Calculate Cross Entropy Loss for EACH sample individually first!
            loss_per_sample = -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(dim=1)
            
            # Multiply by Importance Sampling Weights to fix sampling bias!
            loss = (loss_per_sample * data.weights.squeeze()).mean()
            
            # Use TD-Error (loss_per_sample) as the new priority for these experiences!
            new_priorities = loss_per_sample.detach().cpu().numpy()
            rb.update_priorities(data.indices, new_priorities)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                # BUGFIX: Rainbow generates probability distributions (PMFs) via its Distributional layers instead of scalar Q-Values.
                # To log comparable Q-Values to Tensorboard, we multiply the network support bins by their distribution probability.
                q_values = (old_pmfs * target_network.support).sum(dim=1).mean()
                writer.add_scalar("losses/q_values", q_values.item(), global_step)
                tracker.global_step = global_step
                tracker.log_sps()

        # Hard Network Update
        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    envs.close()
    try:
        tracker.save_checkpoint(agent.state_dict() if "agent" in locals() else q_network.state_dict())
    except:
        pass
    writer.close()