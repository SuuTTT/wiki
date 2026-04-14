# ==============================================================================
# PPO TUTORIAL (Refer to wiki/PPO.md and wiki/Environments.md for theory)
# ==============================================================================

import os
import random
import sys
import time
from dataclasses import dataclass
import tqdm

# Allow importing cleanrl_utils when run from within wiki/learn-cleanrl/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cleanrl')))

# NEW DEPENDENCY: Gymnasium (Standard API for reinforcement learning environments).
import gymnasium as gym

# NEW DEPENDENCY: Numpy (Numerical multidimensional arrays).
import numpy as np

# NEW DEPENDENCY: PyTorch. Our Deep Learning framework.
import torch
import torch.nn as nn
import torch.optim as optim

# NEW DEPENDENCY: tyro (for parsing Command Line Arguments).
import tyro

# NEW: PyTorch Probability Distributions!
# PPO outputs a probability distribution over actions to learn a stochastic policy, instead of deterministic Q-values.
from torch.distributions.categorical import Categorical

from cleanrl_utils.logger import RLTracker
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1                                  # Critical for reproducibility.
    torch_deterministic: bool = True               
    cuda: bool = True                              # Default to Vast.ai GPU!
    track: bool = False                            

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"                    # Classic Control (Discrete).
    total_timesteps: int = 500000                  
    learning_rate: float = 2.5e-4                  
    capture_video: bool = False                     # We enable video capture by default
    
    num_envs: int = 4                              # Number of parallel environments holding different episodes.
    num_steps: int = 128                           # Rollout length per env per update. Data equals num_envs * num_steps.
    
    anneal_lr: bool = True                         # Whether to linearly decay the learning rate.
    
    # THEORY: Advantage Estimation (wiki/PPO.md - Section 2)
    gamma: float = 0.99                            # Discount factor.
    gae_lambda: float = 0.95                       # Lambda parameter for Generalized Advantage Estimation (GAE).
    
    # THEORY: PPO Off-Policy Epochs
    num_minibatches: int = 4                       # How many minibatches to split the rollout into for updates.
    update_epochs: int = 4                         # How many epochs to iterate over the collected rollout data.
    
    # THEORY: The Clipped Surrogate Objective (wiki/PPO.md - Section 3)
    norm_adv: bool = True                          # Whether to normalize advantages before computing loss.
    clip_coef: float = 0.2                         # The clipping coefficient for the policy ratio (epsilon).
    clip_vloss: bool = True                        # Whether to clip the value function loss, similar to the policy.
    
    ent_coef: float = 0.01                         # Entropy coefficient (encourages exploration).
    vf_coef: float = 0.5                           # Value function loss coefficient.
    max_grad_norm: float = 0.5                     # Gradient clamping max norm.
    target_kl: float = None                        # Optional: Early stop update epoch if KL divergence exceeds target.

    # To be computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
    """Sets up the gymnasium environment with video and statistics wrappers."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", lambda x: x % 50 == 0)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# ------------------------------------------------------------------------------
# Neural Network Initializations
# ------------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Orthogonal Initialization: Often outperforms default Xavier/Kaiming in RL.
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ------------------------------------------------------------------------------
# Actor-Critic Architecture (wiki/PPO.md - Section 1)
# ------------------------------------------------------------------------------
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        # CRITIC: Learns the "Value" (expected return) of the given state. Output size is 1!
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(), # Tanh is empirically better in PPO than ReLU.
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0), # Output 1 scalar value.
        )
        
        # ACTOR: Learns the "Policy" (the probability of picking an action). Output size is `num_actions`.
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # Crucial: std=0.01 initializes the weights very small so initial action probs are near uniform (50/50).
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01), 
        )

    def get_value(self, x):
        """Pass states forward and return Critic values."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Pass states forward, return chosen actions, its log probability, entropy, and the Critic's value."""
        logits = self.actor(x) 
        
        # Create a PyTorch Categorical Distribution using the Actor's raw outputs (logits).
        probs = Categorical(logits=logits)
        
        # Sample an action probabilistically during rollout.
        if action is None:
            action = probs.sample()
            
        # Return Action, Log Probability (required for policy ratio), Entropy (for exploration), and the expected Value.
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Calculate batch numbers based on num_envs and num_steps!
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    tracker = RLTracker(args.exp_name, args.seed)
    writer = tracker.writer

    # Set seeds & PyTorch CUDA settings
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Training on: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ==============================================================================
    # PREALLOCATE TENSORS FOR ON-POLICY ROLLOUTS
    # PPO collects a batch of experiences locally (On-Policy), trains on it, then overwrites it.
    # ==============================================================================
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # THE TRAINING LOOP
    progress_bar = tqdm.tqdm(range(1, args.num_iterations + 1), desc="PPO Training", leave=True)
    for iteration in progress_bar:
        # 1. OPTIONAL: Anneal Learning Rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ==============================================================================
        # ROLLOUT PHASE: Collect {args.num_steps} steps of interaction from the environment.
        # ==============================================================================
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                # Ask agent for an action and value.
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            # Save data locally
            actions[step] = action
            logprobs[step] = logprob

            # Apply action to environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "_episode" in infos:
                for idx, d in enumerate(infos["_episode"]):
                    if d:
                        r = infos["episode"]["r"][idx].item() if hasattr(infos["episode"]["r"][idx], "item") else infos["episode"]["r"][idx]
                        l = infos["episode"]["l"][idx].item() if hasattr(infos["episode"]["l"][idx], "item") else infos["episode"]["l"][idx]
                        tracker.log_episode(r, l)
                        if 'progress_bar' in locals():
                            progress_bar.set_postfix(episodic_return=f"{r:.2f}")

        with torch.no_grad():
            # Get the expected value of the current state before the next episode rollouts.
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            # Backwards loop through the collected rollouts.
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    
                # delta = TD error (Reward + Gamma * Next State Value) - Current Value
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                
                # Apply GAE smoothing calculation using Lambda
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                
            returns = advantages + values

        # Flatten the batch to pass back through the optimizer.
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ==============================================================================
        # TRAINING PHASE: Optimizing the Policy and Value Network.
        # ==============================================================================
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds) # Shuffle data
            
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Run data back through current network.
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                
                # Calculate ratio: pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t) 
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # Normalizing advantages at the minibatch level stabilizes learning drastically.
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # ==============================================================================
                # POLICY LOSS (wiki/PPO.md Section 3)
                # ==============================================================================
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # Take the max (negative minimum) to establish the clipped surrogate objective!
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ==============================================================================
                # VALUE LOSS
                # ==============================================================================
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # ==============================================================================
                # TOTAL LOSS = Policy - Entropy + Critic
                # ==============================================================================
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # Clip high gradients!
                optimizer.step()

        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {
                "value_loss": v_loss.item(), 
                "policy_loss": pg_loss.item(), 
                "entropy": entropy_loss.item()
            })
            tracker.global_step = global_step
        tracker.log_sps()

    envs.close()
    try:
        tracker.save_checkpoint(agent.state_dict() if "agent" in locals() else q_network.state_dict())
    except:
        pass
    writer.close()
