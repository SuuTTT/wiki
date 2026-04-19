import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os

# --- Environment Setup (8x8 FrozenLake with explicit layout) ---
def make_env():
    return gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)

# --- Graph Tracker (SIT) ---
class GraphTracker:
    def __init__(self, num_states: int):
        self.num_states = num_states
        self.adj_matrix = np.zeros((num_states, num_states))
        self.clusters = {i: 0 for i in range(num_states)}
        
    def update_edge(self, s, s_next):
        self.adj_matrix[s, s_next] += 1
        
    def run_sit_optimization(self):
        G = nx.from_numpy_array(self.adj_matrix)
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            self.clusters = partition
            return len(set(partition.values()))
        return 1

# --- Worker (PPO-based) ---
class WorkerActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, num_clusters):
        super(WorkerActorCritic, self).__init__()
        self.affine = nn.Linear(num_inputs + num_clusters, 128)
        self.action_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, goal_cluster_onehot):
        x = torch.cat([x, goal_cluster_onehot], dim=-1)
        x = F.relu(self.affine(x))
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)

# --- Manager Actor-Critic ---
class ManagerActorCritic(nn.Module):
    def __init__(self, num_states, num_clusters):
        super(ManagerActorCritic, self).__init__()
        self.affine = nn.Linear(num_states, 64)
        self.goal_head = nn.Linear(64, num_clusters)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.affine(x))
        return F.softmax(self.goal_head(x), dim=-1), self.value_head(x)

# --- Agent Implementation ---
class SITDirector:
    def __init__(self, env, max_clusters=16):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.max_clusters = max_clusters
        
        self.graph_tracker = GraphTracker(self.num_states)
        self.worker = WorkerActorCritic(self.num_states, self.num_actions, self.max_clusters)
        self.manager = ManagerActorCritic(self.num_states, self.max_clusters)
        
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=1e-3)
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=1e-3)

    def to_one_hot(self, s, n):
        arr = torch.zeros(n)
        arr[s] = 1.0
        return arr

    def train_worker(self, rewards, values, log_probs):
        if not rewards: return
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.95 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze()
        if values.dim() == 0: values = values.unsqueeze(0)
        log_probs = torch.stack(log_probs)
        advantage = returns - values.detach()
        loss = -(log_probs * advantage).mean() + 0.5 * F.mse_loss(values, returns)
        self.worker_optimizer.zero_grad()
        loss.backward()
        self.worker_optimizer.step()

    def train_manager(self, rewards, values, log_probs):
        if not rewards: return
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze()
        if values.dim() == 0: values = values.unsqueeze(0)
        log_probs = torch.stack(log_probs)
        advantage = returns - values.detach()
        loss = -(log_probs * advantage).mean() + 0.5 * F.mse_loss(values, returns)
        self.manager_optimizer.zero_grad()
        loss.backward()
        self.manager_optimizer.step()

    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        total_extr_reward = 0
        
        w_rewards, w_vals, w_logprobs = [], [], []
        m_rewards, m_vals, m_logprobs = [], [], []

        state_tensor = self.to_one_hot(state, self.num_states)
        goal_probs, manager_val = self.manager(state_tensor)
        goal_dist = Categorical(goal_probs)
        target_cl = goal_dist.sample()

        steps = 0
        while not done and steps < 200:
            curr_cl = self.graph_tracker.clusters.get(state, 0)
            goal_target_onehot = torch.zeros(self.max_clusters)
            goal_target_onehot[target_cl] = 1.0
            
            probs, val = self.worker(state_tensor, goal_target_onehot)
            m = Categorical(probs)
            action = m.sample()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            steps += 1
            
            self.graph_tracker.update_edge(state, next_state)
            next_cl = self.graph_tracker.clusters.get(next_state, 0)
            
            intrinsic_reward = 1.0 if next_cl == target_cl.item() and curr_cl != target_cl.item() else -0.01
            
            w_rewards.append(intrinsic_reward)
            w_vals.append(val)
            w_logprobs.append(m.log_prob(action))
            
            if next_cl != curr_cl or intrinsic_reward > 0 or done:
                m_rewards.append(reward + (0.5 if intrinsic_reward > 0 else 0))
                m_vals.append(manager_val)
                m_logprobs.append(goal_dist.log_prob(target_cl))

                next_state_tensor = self.to_one_hot(next_state, self.num_states)
                goal_probs, manager_val = self.manager(next_state_tensor)
                goal_dist = Categorical(goal_probs)
                target_cl = goal_dist.sample()
                state_tensor = next_state_tensor
            
            state = next_state
            state_tensor = self.to_one_hot(state, self.num_states)
            total_extr_reward += reward
            
        self.train_worker(w_rewards, w_vals, w_logprobs)
        self.train_manager(m_rewards, m_vals, m_logprobs)
        return total_extr_reward

# --- Main Experiment Loop ---
def run_experiment(episodes=1000):
    env = make_env()
    sit_agent = SITDirector(env)
    
    results = []
    print("Starting SIT-Director Experiment...")
    
    for ep in range(episodes):
        rew = sit_agent.run_episode()
        
        if (ep + 1) % 100 == 0:
            n_c = sit_agent.graph_tracker.run_sit_optimization()
            print(f"Episode {ep+1} | Reward: {rew:.2f} | Clusters: {n_c}")
            
            # Record metrics
            results.append({
                "Episode": ep + 1,
                "Reward": rew,
                "Clusters": n_c
            })
            
            # Visualize abstract topology
            grid = np.zeros((8, 8))
            for state, cluster in sit_agent.graph_tracker.clusters.items():
                r, c = divmod(state, 8)
                grid[r, c] = cluster
            
            plt.figure(figsize=(6, 5))
            plt.imshow(grid, cmap='tab20')
            plt.title(f"SIT Abstract Discovery (Ep {ep+1})")
            plt.colorbar(label="Macro-state ID")
            plt.savefig(f"sit_topology_ep{ep+1}.png")
            plt.close()

    # Final Report Creation
    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    
    # Summary Figure
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df["Episode"], df["Reward"], marker='o')
    plt.title("Extrinsic Reward (Goal Reached)")
    plt.ylabel("Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(df["Episode"], df["Clusters"], marker='x', color='r')
    plt.title("Discovered Macro-states (SIT)")
    plt.ylabel("Num Clusters")
    
    plt.tight_layout()
    plt.savefig("experiment_summary.png")
    print("Experiment Complete. Results saved to experiment_results.csv and experiment_summary.png.")

if __name__ == "__main__":
    run_experiment(1500)
