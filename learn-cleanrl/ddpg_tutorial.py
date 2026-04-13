# ==============================================================================
# DDPG TUTORIAL (Refer to wiki/DDPG.md and wiki/ContinuousControl.md for theory)
# ==============================================================================

import os
import random
import sys
import time
from dataclasses import dataclass
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cleanrl')))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = True

    # Algorithm specific arguments
    # THEORY: We are no longer using Discrete CartPole. 
    # Pendulum-v1 is a classic continuous control environment (applying torque to swing a pendulum up).
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    
    # Just like DQN: Off-policy method so we need a replay buffer to store generic experiences.
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    batch_size: int = 256
    
    # THEORY: Soft Target Updates (wiki/DDPG.md - Section 4)
    # Instead of updating perfectly every 500 steps, we do a soft interpolation every step by 0.005.
    tau: float = 0.005
    
    # THEORY: Deterministic Exploration Noise (wiki/DDPG.md - Section 1)
    exploration_noise: float = 0.1
    
    learning_starts: int = 25e3
    
    # THEORY: Delayed Policy Updates.
    # The Critic needs to learn to predict Q-Values properly before the Actor can trust it to improve itself.
    # DDPG trains the Critic every step, but delays training the Actor (usually every 2 steps).
    policy_frequency: int = 2

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

# THEORY: The Continuous Critic (wiki/DDPG.md - Section 2)
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # IN CONTINUOUS spaces, the Critic evaluates a (State, Action) PAIR.
        # We concatenate the State Vector AND the Continuous Action Vector as INPUT.
        # It outputs a single expected Q-Value!
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        # Concatenate the State (x) and the Continuous Action (a) at dimension 1
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# THEORY: The Deterministic Actor (wiki/DDPG.md - Section 1)
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Continuous action space Actor.
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        # Outputs an exact Float for every action joint (e.g. 3 joints for Hopper-v4)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        
        # Action Rescaling: Tanh outputs [-1, 1]. If the environment needs [-2, 2], we must scale it!
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Tanh outputs a crisp [-1, 1] constraint.
        x = torch.tanh(self.fc_mu(x))
        # Multiply by the scale + bias so the actual output perfectly fits the gym environment bounds!
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    writer = SummaryWriter(f"runs/{run_name}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    # DDPG typically does not vectorize environments.
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    
    # Assert we are specifically using a Box (Continuous) action space!
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action spaces are supported in DDPG"

    # Initialize Networks
    actor = Actor(envs).to(device)
    # The Primary Critic (Q1)
    qf1 = QNetwork(envs).to(device)
    
    # The Target Networks (Both Actor AND Critic need Target networks connected by Soft-Updates)!
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    
    # Copy exact weights on init.
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # Initialize Replay Buffer explicitly for Box action spaces (Continuous floats)
    envs.single_observation_space.dtype = np.float32
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
    progress_bar = tqdm.tqdm(range(args.total_timesteps), desc="DDPG Training", leave=True)
    for global_step in progress_bar:
        
        # 1. EXPLORATION
        if global_step < args.learning_starts:
            # We start by acting entirely randomly to fill the replay buffer with diverse generic transitions.
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                # Ask the actor what deterministic action to take.
                actions = actor(torch.Tensor(obs).to(device))
                # Add Gaussian noise (N(0, sigma)) specifically for DDPG exploration.
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                # Clip the action so the noise doesn't push the float past the environment bounds (e.g. [-1.0, 1.0]).
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # 2. ENVIRONMENT STEP
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log Episode
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    progress_bar.set_postfix(episodic_return=info['episode']['r'][0] if isinstance(info['episode']['r'], np.ndarray) else info['episode']['r'])
                    tqdm.tqdm.write(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    break

        # Save to buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # ==============================================================================
        # TRAINING (OFF-POLICY DDPG)
        # ==============================================================================
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # --- CRITIC TRAINING ---
            with torch.no_grad():
                # Ask the Target Actor what action it WOULD take in the NEXT state.
                next_state_actions = target_actor(data.next_observations)
                # Feed both the next state AND next predicted action into the Target Critic.
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                # Bellman Equation: TD_Target = r + gamma * Target_Q(s', a')
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            # Ask the Primary Critic what it thinks of the CURRENT state/action that we ACTUALLY took.
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            
            # Critic Loss is just Mean Squared Error between Bellman target and our prediction.
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # Optimize the Critic!
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            # --- ACTOR TRAINING (wiki/DDPG.md Section 3) ---
            # Delay the actor updates so the Critic has time to stabilize its Q-values.
            if global_step % args.policy_frequency == 0:
                # DDPG "Policy Gradient" is literally just maximizing the Critic output:
                # What deterministic action does the Actor predict for the current state?
                predicted_action = actor(data.observations)
                
                # What is the Critic's Q-Value evaluated on that deterministic action?
                # Our gradient is the negative of the Q-Value (so taking a gradient step maximizes Q):
                actor_loss = -qf1(data.observations, predicted_action).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --- SOFT TARGET UPDATE (wiki/DDPG.md Section 4) ---
                # Slowly blend weights over tau (0.005) instead of a hard copy.
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
