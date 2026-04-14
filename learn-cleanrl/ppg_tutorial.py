# ==============================================================================
# PPG TUTORIAL (Refer to wiki/PPG.md for theory)
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
from cleanrl_utils.logger import RLTracker
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "ppg_tutorial"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 5e-4
    num_envs: int = 8
    num_steps: int = 256
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Phase 1: Policy Phase specifics
    e_policy: int = 2 # Epochs for policy
    num_minibatches: int = 4
    n_iteration: int = 16 # After how many policy phases do we run the Aux phase?
    
    # Phase 2: Auxiliary Phase specifics
    e_auxiliary: int = 6 # Epochs for aux phase
    beta_clone: float = 1.0 # KL Divergence cloning coefficient
    num_aux_minibatches: int = 4
    
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPGAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        # 1. The Actor Network (Disjoint from the Critic's main features)
        self.actor_network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        
        # THEORY: The Auxiliary Value Head (wiki/PPG.md)
        # This head lives on the ACTOR's feature extractor, but tries to predict the CRITIC's values.
        self.aux_critic = layer_init(nn.Linear(64, 1), std=1)
        
        # 2. The Critic Network (Completely separate from the Actor)
        self.critic_network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_action_and_value(self, x, action=None):
        """Standard Policy phase calls"""
        hidden = self.actor_network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        critic_hidden = self.critic_network(x)
        value = self.critic(critic_hidden)
        
        return action, probs.log_prob(action), probs.entropy(), value

    def get_action(self, x):
        """Get pure actions"""
        hidden = self.actor_network(x)
        logits = self.actor(hidden)
        return Categorical(logits=logits)

    def get_value(self, x):
        """Get pure critic values"""
        hidden = self.critic_network(x)
        return self.critic(hidden)

    def get_pi_value_and_action(self, x, action=None):
        """THEORY: The Auxiliary Phase call"""
        # We pass state through the ACTOR network
        hidden = self.actor_network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            
        # We predict the value using the ACTOR's auxiliary critic head
        aux_value = self.aux_critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), aux_value


if __name__ == "__main__":
    args = tyro.cli(Args)
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = args.total_timesteps // batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    tracker = RLTracker(args.exp_name, args.seed)
    writer = tracker.writer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    agent = PPGAgent(envs).to(device)
    
    # Separate optimizers for the pure networks
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # ------------------------------------------------------------------------------
    # AUXILIARY BUFFER (To hold 16 policy phases worth of data)
    # ------------------------------------------------------------------------------
    aux_obs = torch.zeros((args.num_steps, args.num_envs, args.n_iteration) + envs.single_observation_space.shape, device=device)
    aux_returns = torch.zeros((args.num_steps, args.num_envs, args.n_iteration), device=device)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    progress_bar = tqdm.tqdm(range(1, num_iterations + 1), desc="PPG Training")

    for iteration in progress_bar:
        # 1. ROLLOUTS
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs_np).to(device), torch.Tensor(next_done_np).to(device)

            if "_episode" in infos:
                for idx, d in enumerate(infos["_episode"]):
                    if d:
                        r = infos["episode"]["r"][idx].item() if hasattr(infos["episode"]["r"][idx], "item") else infos["episode"]["r"][idx]
                        l = infos["episode"]["l"][idx].item() if hasattr(infos["episode"]["l"][idx], "item") else infos["episode"]["l"][idx]
                        tracker.log_episode(r, l)
                        if 'progress_bar' in locals():
                            progress_bar.set_postfix(episodic_return=f"{r:.2f}")

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

        # Store data for the Auxiliary Phase
        aux_idx = (iteration - 1) % args.n_iteration
        aux_obs[:, :, aux_idx] = obs
        aux_returns[:, :, aux_idx] = returns

        # ------------------------------------------------------------------------------
        # PHASE 1: THE POLICY PHASE
        # ------------------------------------------------------------------------------
        b_inds = np.arange(batch_size)
        for epoch in range(args.e_policy):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                
                # We optimize both, but notice there's no feature interference because the networks are disjoint!
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ------------------------------------------------------------------------------
        # PHASE 2: THE AUXILIARY PHASE (Feature Sharing Phase)
        # ------------------------------------------------------------------------------
        if iteration % args.n_iteration == 0:
            
            # 1. Generate "True Value" Targets from the massively accurate Disjoint Critic
            # Provide the "Old Policy" logic to calculate KL Divergence against
            aux_b_obs = aux_obs.reshape((-1,) + envs.single_observation_space.shape)
            aux_b_returns = aux_returns.reshape(-1)
            
            with torch.no_grad():
                aux_b_values = agent.get_value(aux_b_obs)
                # Save the precise output of the Actor as "historical ground truth"
                aux_b_old_logits = agent.get_action(aux_b_obs).logits

            aux_batch_size = aux_b_obs.shape[0]
            aux_minibatch_size = int(aux_batch_size // args.num_aux_minibatches)
            aux_b_inds = np.arange(aux_batch_size)

            for epoch in range(args.e_auxiliary):
                np.random.shuffle(aux_b_inds)
                for start in range(0, aux_batch_size, aux_minibatch_size):
                    end = start + aux_minibatch_size
                    mb_inds = aux_b_inds[start:end]

                    # 2. Get the Actor's auxiliary value prediction, AND its new logits
                    _, newlogprob, entropy, new_aux_value = agent.get_pi_value_and_action(aux_b_obs[mb_inds], b_actions.reshape(-1)[mb_inds] if mb_inds.max() < b_actions.reshape(-1).shape[0] else None)
                    new_dist = agent.get_action(aux_b_obs[mb_inds])
                    new_logits = new_dist.logits
                    
                    # 3. Calculate "Policy Cloning" KL Divergence
                    # Force the new policy probabilities to mathematically match the old ones!
                    old_dist = Categorical(logits=aux_b_old_logits[mb_inds])
                    kl_loss = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

                    # 4. Auxiliary Value Loss (Train Actor's Aux Head to match the main Critic)
                    v_loss_aux = 0.5 * ((new_aux_value.view(-1) - aux_b_returns[mb_inds]) ** 2).mean()

                    # 5. Critic Value Loss (Main critic continues to refine its MSE against returns)
                    v_loss_main = 0.5 * ((agent.get_value(aux_b_obs[mb_inds]).view(-1) - aux_b_returns[mb_inds]) ** 2).mean()

                    # Combine all losses
                    # The Actor's gradient forces its features to learn to predict Value through `v_loss_aux`
                    # BUT `kl_loss` prevents that gradient from disrupting behavior.
                    loss = v_loss_main + v_loss_aux + args.beta_clone * kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
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