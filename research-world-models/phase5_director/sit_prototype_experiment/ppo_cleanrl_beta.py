import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = "ppo_beta_cleanrl"
    seed: int = 2 # Change seed
    torch_deterministic: bool = True
    cuda: bool = True
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 2000000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # Standard Wrappers
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_in, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_in, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 2 * n_out), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # DETECT NAN IN INPUT
        if torch.isnan(x).any():
            print("WARNING: NAN IN INPUT OBS")
            x = torch.nan_to_num(x)
            
        latent = self.actor(x)
        # Softplus + safe offset
        alpha_beta = torch.nn.functional.softplus(latent) + 1.01
        alpha, beta = torch.chunk(alpha_beta, 2, dim=-1)
        
        # FINAL SAFETY CHECK
        if torch.isnan(alpha).any() or torch.isnan(beta).any():
            print("CRITICAL: NAN IN ALPHA/BETA")
            alpha = torch.nan_to_num(alpha, nan=2.0)
            beta = torch.nan_to_num(beta, nan=2.0)
            
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

if __name__ == "__main__":
    args = Args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__beta_final__{args.seed}__{int(time.time())}"
    
    writer = SummaryWriter(f"logs/continuous_study/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = make_env(args.env_id, args.seed)()
    
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)
    low = torch.tensor(env.action_space.low).to(device)
    high = torch.tensor(env.action_space.high).to(device)

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, act_dim)).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs_np).to(device).unsqueeze(0)
    next_done = 0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action.squeeze(0)
            logprobs[step] = logprob

            env_action = low + action.squeeze(0) * (high - low)
            next_obs_np, reward, term, trunc, info = env.step(env_action.cpu().numpy())
            next_done = term or trunc
            rewards[step] = torch.tensor(reward).to(device)
            
            if next_done:
                next_obs_np, _ = env.reset()
                if "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

            next_obs = torch.Tensor(next_obs_np).to(device).unsqueeze(0)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
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

        # Flattened batch optimization
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[mb_inds], actions[mb_inds])
                ratio = (newlogprob - logprobs[mb_inds]).exp()

                mb_advantages = advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration} | Global Step {global_step} | SPS {int(global_step / (time.time() - start_time))}")

    writer.close()
