# ==============================================================================
# DQN TUTORIAL (Refer to wiki/DQN.md and wiki/Environments.md for theory)
# ==============================================================================

import os
import random
import time
from dataclasses import dataclass
import tqdm

# NEW DEPENDENCY: Gymnasium. This is the standard API for reinforcement learning environments.
# It provides `env.reset()` to start an episode, and `env.step(action)` to take an action.
import gymnasium as gym

# NEW DEPENDENCY: Numpy. Used for handling non-GPU arrays and numerical operations.
import numpy as np

# NEW DEPENDENCY: PyTorch. Our Deep Learning framework.
import torch
import torch.nn as nn              # Neural network layers (Linear, ReLU, etc.)
import torch.nn.functional as F    # Functional math operations (MSE loss, etc.)
import torch.optim as optim        # Optimizers (Adam, SGD, etc.)

# NEW DEPENDENCY: Tyro. A modern CLI parser. It reads the `Args` dataclass and 
# lets you pass variables from the terminal (e.g. `python dqn.py --learning-rate 0.001`).
import tyro

# NEW DEPENDENCY: Tensorboard. Used for logging scalars (like reward, loss) to view in a dashboard.
from cleanrl_utils.logger import RLTracker
from torch.utils.tensorboard import SummaryWriter

# LOCAL: The replay buffer. Explained in wiki/DQN.md Section 2.
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    # --- Experiment Configuration ---
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1                          # Critical for reproducibility.
    torch_deterministic: bool = True       # Ensures GPU operations are deterministic where possible.
    cuda: bool = True                      # We default to using the Vast.ai GPU!
    capture_video: bool = False             # We enable video capture!
    
    # --- Algorithm Specifics (DQN hyperparameters) ---
    env_id: str = "CartPole-v1"            # Starting with the simplest environment! (See wiki/Environments.md)
    total_timesteps: int = 500000          # Total frames the agent will experience.
    learning_rate: float = 2.5e-4          # Step size for the Adam Optimizer.
    num_envs: int = 1                      # Vectorized environments (1 for basic DQN).
    
    # THEORY: Replay Buffer (wiki/DQN.md - Section 2)
    buffer_size: int = 10000               # How many past transitions to remember.
    
    # THEORY: The Bellman Equation (wiki/DQN.md - Section 1)
    gamma: float = 0.99                    # The discount factor (importance of future rewards).
    
    # THEORY: Target Networks (wiki/DQN.md - Section 3)
    tau: float = 1.0                       # Hard update rate for target network.
    target_network_frequency: int = 500    # How often (in steps) we copy weights to the Target Network.
    
    batch_size: int = 128                  # How many transitions to sample from the buffer for one train step.
    
    # Epsilon-Greedy Exploration: Start with 1.0 (100% random), decay to 0.05 (5% random).
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5      # Takes 50% of total timesteps to reach `end_e`.
    
    learning_starts: int = 10000           # Wait until 10k random steps to fill buffer before training.
    train_frequency: int = 10              # Train the network every 10 steps.


def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Creates the environment. Gymnasium environments require wrappers to automatically
    record episode returns (RecordEpisodeStatistics) or video.
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", lambda x: x % 50 == 0)
        else:
            env = gym.make(env_id)
        
        # RecordEpisodeStatistics automatically adds the 'episode' key to `info` when an episode ends.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


# ------------------------------------------------------------------------------
# Neural Network Architecture
# ------------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # A simple Multi-Layer Perceptron (MLP).
        # Inputs: State (e.g. 4 for CartPole).
        # Outputs: Expected Return for each Action (e.g. 2 for CartPole).
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        # PyTorch forwards pass. Input state -> output Q-values for all actions.
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Calculates epsilon for epsilon-greedy exploration."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Setup Tensorboard
    tracker = RLTracker(args.exp_name, args.seed)
    writer = tracker.writer

    # Set seeds for Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Setup Device (Use GPU if available on Vast.ai docker)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Training on: {device}")

    # Setup Environment
    # SyncVectorEnv helps run multiple environments in parallel, but here we just use 1.
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    # Initialize the Primary Q-Network and its optimizer.
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    
    # INITIALIZE TARGET NETWORK (wiki/DQN.md Section 3)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict()) # Copy weights exactly.

    # Initialize Replay Buffer (wiki/DQN.md Section 2)
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    # ==============================================================================
    # THE TRAINING LOOP
    # ==============================================================================
    progress_bar = tqdm.tqdm(range(args.total_timesteps), desc="DQN Training", leave=True)
    for global_step in progress_bar:
        # 1. EXPLORATION VS EXPLOITATION
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            # Explore: Take a random action.
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Exploit: Ask the Q-network for the best action.
            # Convert observation to PyTorch tensor -> run through network -> get `argmax` action.
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # 2. STEP THE ENVIRONMENT
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log rewards automatically collected by the RecordEpisodeStatistics wrapper.
        if "_episode" in infos:
                for idx, d in enumerate(infos["_episode"]):
                    if d:
                        r = infos["episode"]["r"][idx].item() if hasattr(infos["episode"]["r"][idx], "item") else infos["episode"]["r"][idx]
                        l = infos["episode"]["l"][idx].item() if hasattr(infos["episode"]["l"][idx], "item") else infos["episode"]["l"][idx]
                        tracker.log_episode(r, l)
                        if 'progress_bar' in locals():
                            progress_bar.set_postfix(episodic_return=f"{r:.2f}")

        real_next_obs = next_obs.copy()
        # Edge case: If the environment forcefully truncates (time limit), grab the real final state.
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
                
        # Store transition (s, a, r, s', done)
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # Move to the next state!
        obs = next_obs

        # ==============================================================================
        # OPTIMIZATION STEP
        # ==============================================================================
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            # Sample a batch of random transitions to break correlation (wiki/DQN.md Section 2)
            data = rb.sample(args.batch_size)
            
            # --- BELLMAN EQUATION ---
            # We don't want gradients flowing into the target network.
            with torch.no_grad():
                # Find the maximum Q-value in the next state using the TARGET network.
                target_max, _ = target_network(data.next_observations).max(dim=1)
                
                # Formula: TD_Target = reward + gamma * max_a' Q(s', a') * (1 - done)
                # If done=1, the episode is over, so the target is just the reward.
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            
            # Get the Q-values of the *Primary* network for the actions we actually took.
            # .gather(1, data.actions) basically picks out the Q-value for the action we played.
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            
            # Compute Mean Squared Error (MSE) loss between prediction and target.
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

            # PyTorch Magic: Zero gradients, backpropagate, step optimizer.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 4. UPDATE TARGET NETWORK (wiki/DQN.md Section 3)
        if global_step % args.target_network_frequency == 0:
            for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                # Copying weights from the primary network to the target network.
                target_param.data.copy_(args.tau * q_param.data + (1.0 - args.tau) * target_param.data)

    envs.close()
    try:
        tracker.save_checkpoint(agent.state_dict() if "agent" in locals() else q_network.state_dict())
    except:
        pass
    writer.close()
