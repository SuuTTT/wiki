import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import networkx as nx
import community as community_louvain
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from sit_tdmpc_model import SITWorldModel, Config

class MPPIPlanner:
    """
    Simplified Model Predictive Path Integral (MPPI) for latent space planning.
    """
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.horizon = 5 # SIT-Macro Horizon
        self.num_samples = 32

    @torch.no_grad()
    def plan(self, z):
        # z: [1, latent_dim]
        best_reward = -float('inf')
        best_action_seq = None
        
        # --- Jumpy-MPC Strategy ---
        # If SIT tree is available, we look for a "Subgoal" in the next module 
        # instead of just random shooting.
        target_subgoal = None
        if self.model.sit_tree.tree_structure is not None:
             # Find a node in a different cluster that has been frequently visited
             target_subgoal = self.model.get_abstract_subgoal(z)

        for _ in range(self.num_samples):
            actions = torch.randint(0, self.cfg.action_dim, (self.horizon,))
            curr_z = z
            total_reward = 0
            for a in actions:
                a_onehot = F.one_hot(a, num_classes=self.cfg.action_dim).float().unsqueeze(0).to(z.device)
                curr_z = self.model.next(curr_z, a_onehot)
                
                # Reward = Task Reward + SIT Proximity Reward
                reward = self.model._reward(curr_z).item()
                if target_subgoal is not None:
                    # SIT intrinsic: Reward for reaching the predicted boundary state
                    dist = torch.norm(curr_z - target_subgoal)
                    reward += 0.5 * torch.exp(-dist) # Gaussian SIT-potential
                
                total_reward += reward
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_action_seq = actions
        
        return best_action_seq[0] if best_action_seq is not None else torch.tensor(0)

class SIT_TD_MPC_Trainer:
    def __init__(self, env_name="FrozenLake-v1"):
        self.cfg = Config()
        self.cfg.obs_dim = 64 # One-hot 8x8
        self.cfg.action_dim = 4
        self.cfg.latent_dim = 32
        
        self.env = gym.make(env_name, is_slippery=False, render_mode=None, desc=[
            "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFVF", "FFFHFFFG"
        ])
        self.model = SITWorldModel(self.cfg)
        self.planner = MPPIPlanner(self.model, self.cfg)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=5000)
        self.writer = SummaryWriter(log_dir="/workspace/runs/SIT-TDMPC-8x8")
        self.global_step = 0

    def preprocess_obs(self, obs):
        # Convert discrete index to one-hot vector
        one_hot = np.zeros(64)
        one_hot[obs] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0)

    def train_step(self):
        if len(self.replay_buffer) < 128: return
        
        # Basic TD-MPC loss sample
        # In real version, this uses full batch and temporal consistency
        idx = np.random.choice(len(self.replay_buffer))
        obs, action, reward, next_obs, done = self.replay_buffer[idx]
        
        z = self.model.encode(obs)
        a_onehot = F.one_hot(torch.tensor(action), num_classes=self.cfg.action_dim).float().unsqueeze(0)
        next_z_pred = self.model.next(z, a_onehot)
        next_z_real = self.model.encode(next_obs)
        
        # Consistency Loss
        loss = F.mse_loss(next_z_pred, next_z_real.detach())
        # Reward Loss
        reward_pred = self.model._reward(next_z_pred)
        reward_loss = F.mse_loss(reward_pred, torch.tensor([[reward]]).float())
        loss += reward_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_step % 10 == 0:
            self.writer.add_scalar("train/total_loss", loss.item(), self.global_step)
            self.writer.add_scalar("train/reward_loss", reward_loss.item(), self.global_step)
        
        return loss.item()

    def run(self, num_episodes=500):
        print(f"Starting SIT-TD-MPC on 8x8 GridWorld...")
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            obs = self.preprocess_obs(obs)
            total_reward = 0
            done = False
            
            while not done:
                z = self.model.encode(obs)
                action = self.planner.plan(z).item()
                
                next_obs_raw, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = self.preprocess_obs(next_obs_raw)
                done = terminated or truncated
                
                # --- SIT Intrinsic Reward Logic ---
                intrinsic_reward = 0
                if self.model.sit_tree.tree_structure is not None:
                    # Current and Next state IDs (quantized to buffer index)
                    curr_id = len(self.model.state_buffer) - 1
                    next_id = len(self.model.state_buffer)
                    if curr_id in self.model.sit_tree.tree_structure and next_id in self.model.sit_tree.tree_structure:
                        # Reward for crossing module boundaries (moving to a different parent in tree)
                        if self.model.sit_tree.tree_structure[curr_id] != self.model.sit_tree.tree_structure[next_id]:
                            intrinsic_reward = 0.1 # "Boundary Crossing" curiosity
                # ----------------------------------

                self.replay_buffer.append((obs, action, reward + intrinsic_reward, next_obs, done))
                # Update latent graph for SIT
                self.model.state_buffer.append(obs.numpy().flatten())
                
                obs = next_obs
                total_reward += reward
                self.global_step += 1
                self.train_step()

            self.writer.add_scalar("train/episode_reward", total_reward, ep)
            if ep % 50 == 0:
                self.model.update_sit_abstractions()
                print(f"Episode {ep} | Reward: {total_reward} | Buffer: {len(self.replay_buffer)}")

if __name__ == "__main__":
    trainer = SIT_TD_MPC_Trainer()
    trainer.run(num_episodes=101)
