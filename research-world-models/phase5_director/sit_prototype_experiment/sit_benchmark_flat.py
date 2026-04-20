import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import community as community_louvain
import time
import os
from collections import deque

# 4x4 Map
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
def make_env(): return gym.make("FrozenLake-v1", desc=MAP, is_slippery=False)

class GraphTracker:
    def __init__(self, num_states: int):
        self.num_states = num_states
        self.adj_matrix = np.zeros((num_states, num_states))
        self.clusters = {i: 0 for i in range(num_states)}
    def update_edge(self, s, s_next): self.adj_matrix[s, s_next] += 1
    def run_sit_optimization(self):
        G = nx.from_numpy_array(self.adj_matrix)
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G)
            self.clusters = partition
            return len(partition)
        return 1

class AC(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_out), nn.Softmax(-1))
        self.val = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 1))

def run_experiment(mode="hierarchical", episodes=1500):
    run_name = f"benchmark_{mode}_{int(time.time())}"
    writer = SummaryWriter(f"logs/{run_name}")
    print(f"Starting {mode} experiment...")
    
    env = make_env()
    graph = GraphTracker(16)
    
    # Hierarchical has Manager + Worker (input=16+8)
    # Flat has only Worker (input=16)
    if mode == "hierarchical":
        worker = AC(16 + 8, 4)
        manager = AC(16, 8)
        m_opt = optim.Adam(manager.parameters(), 5e-4)
    else:
        worker = AC(16, 4)
        manager = None
        
    w_opt = optim.Adam(worker.parameters(), 1e-3)
    recent_rewards = deque(maxlen=100)
    
    for ep in range(episodes):
        s, _ = env.reset(); done = False; tot_rew = 0; steps = 0
        w_hist, m_hist = [], []
        s_oh = torch.eye(16)[s]
        
        # Initial Manager Goal
        if mode == "hierarchical":
            g_probs, m_val = manager.net(s_oh), manager.val(s_oh)
            dist = Categorical(g_probs); target = dist.sample()
        
        while not done and steps < 100:
            if mode == "hierarchical":
                g_oh = torch.eye(8)[target]
                w_in = torch.cat([s_oh, g_oh])
            else:
                w_in = s_oh
                
            w_probs, w_val = worker.net(w_in), worker.val(w_in)
            
            # Epsilon-greedy (Same for both)
            eps = max(0.1, 0.8 - ep/1000)
            if np.random.rand() < eps: action = torch.tensor(np.random.randint(4))
            else: action = Categorical(w_probs).sample()
            
            ns, r, term, trunc, _ = env.step(action.item())
            done = term or trunc; steps += 1
            graph.update_edge(s, ns)
            
            if mode == "hierarchical":
                ns_cl = graph.clusters.get(ns, 0); curr_cl = graph.clusters.get(s, 0)
                # Intrinsic reward (reaching target cluster)
                ir = 0.2 if ns_cl == target.item() and curr_cl != target.item() else -0.01
                w_hist.append((w_val, Categorical(w_probs).log_prob(action), ir + r))
                
                if ns_cl != curr_cl or ir > 0.1 or done:
                    r_m = float(r) * 10.0 + (1.0 if ir > 0.1 else 0)
                    m_hist.append((m_val, dist.log_prob(target), r_m))
                    s_oh = torch.eye(16)[ns]
                    g_probs, m_val = manager.net(s_oh), manager.val(s_oh)
                    dist = Categorical(g_probs); target = dist.sample()
                else: s_oh = torch.eye(16)[ns]
            else:
                # Flat PPO only sees extrinsic reward
                w_hist.append((w_val, Categorical(w_probs).log_prob(action), float(r)))
                s_oh = torch.eye(16)[ns]
            
            s = ns; tot_rew += r

        # Optimize
        for hist, opt in [(w_hist, w_opt), (m_hist, m_opt if manager else None)]:
            if not hist or not opt: continue
            vals, logs, rews = zip(*hist)
            vals = torch.stack(vals).view(-1)
            R = []; curr_r = 0
            for r_val in reversed(rews):
                curr_r = r_val + 0.99 * curr_r
                R.insert(0, curr_r)
            R = torch.tensor(R, dtype=torch.float32)
            adv = R - vals.detach()
            loss = -(torch.stack(logs) * adv).mean() + 0.1 * F.mse_loss(vals, R)
            opt.zero_grad(); loss.backward(); opt.step()

        if ep % 10 == 0: graph.run_sit_optimization()
        recent_rewards.append(tot_rew)
        avg_rew = np.mean(recent_rewards)
        writer.add_scalar("charts/episodic_return", avg_rew, ep)
        if mode == "hierarchical":
            # Track if intrinsic reward is rising
            writer.add_scalar("charts/intrinsic_reward_sum", sum([h[2] for h in w_hist]), ep)

    writer.close()
    print(f"Finished {mode}. Final AvgRew: {avg_rew:.2f}")

if __name__ == "__main__":
    run_experiment(mode="hierarchical", episodes=1200)
    run_experiment(mode="flat", episodes=1200)
