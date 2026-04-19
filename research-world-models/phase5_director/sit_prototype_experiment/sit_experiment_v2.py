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

# Simplified Map (Deterministic 4x4 for faster proof)
MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

def make_env():
    return gym.make("FrozenLake-v1", desc=MAP, is_slippery=False)

# Graph Tracker (SIT)
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

# Networks
class Worker(nn.Module):
    def __init__(self, n_in, n_out, n_cl):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in+n_cl, 64), nn.ReLU(), nn.Linear(64, n_out), nn.Softmax(-1))
        self.val = nn.Sequential(nn.Linear(n_in+n_cl, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, g):
        cat = torch.cat([x, g], -1)
        return self.net(cat), self.val(cat)

class Manager(nn.Module):
    def __init__(self, n_in, n_cl):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_cl), nn.Softmax(-1))
        self.val = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 1))

# Main Loop
def train(episodes=2000):
    env = make_env()
    n_s, n_a, n_c = 16, 4, 8
    graph = GraphTracker(16)
    worker = Worker(16, n_a, n_c)
    manager = Manager(16, n_c)
    w_opt = optim.Adam(worker.parameters(), 1e-3)
    m_opt = optim.Adam(manager.parameters(), 5e-4)

    results = []
    
    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        tot_rew = 0
        
        w_hist, m_hist = [], []
        s_oh = torch.eye(16)[s]
        g_prob, m_val = manager.net(s_oh), manager.val(s_oh)
        target = Categorical(g_prob).sample()

        while not done:
            curr_cl = graph.clusters.get(s, 0)
            g_oh = torch.eye(n_c)[target]
            probs, val = worker(s_oh, g_oh)
            action = Categorical(probs).sample()
            
            ns, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            graph.update_edge(s, ns)
            ns_cl = graph.clusters.get(ns, 0)
            
            # Intrinsic: Positive for reaching target cluster, otherwise small penalty
            ir = 2.0 if ns_cl == target.item() and curr_cl != target.item() else -0.05
            w_hist.append((val, m.log_prob(action) if 'm' in locals() else Categorical(probs).log_prob(action), ir))
            
            if ns_cl != curr_cl or ir > 0 or done:
                m_hist.append((m_val, Categorical(g_prob).log_prob(target), r + (1.0 if ir > 0 else 0)))
                s_oh = torch.eye(16)[ns]
                g_prob, m_val = manager.net(s_oh), manager.val(s_oh)
                target = Categorical(g_prob).sample()

            s = ns
            s_oh = torch.eye(16)[s]
            tot_rew += r

        # Update
        if len(w_hist) > 0:
            v, lp, r_ = zip(*w_hist)
            v = torch.cat(v).squeeze()
            R = torch.tensor(r_).cumsum(0).flip(0) # Simple returns
            loss = F.mse_loss(v, R) - (torch.stack(lp) * (R - v.detach())).mean()
            w_opt.zero_grad(); loss.backward(); w_opt.step()
        
        if len(m_hist) > 0:
            v, lp, r_ = zip(*m_hist)
            v = torch.cat(v).squeeze()
            if v.dim() == 0: v = v.unsqueeze(0)
            R = torch.tensor(r_).cumsum(0).flip(0)
            loss = F.mse_loss(v, R) - (torch.stack(lp) * (R - v.detach())).mean()
            m_opt.zero_grad(); loss.backward(); m_opt.step()

        if (ep + 1) % 100 == 0:
            c_count = graph.run_sit_optimization()
            results.append({"Episode": ep+1, "Reward": tot_rew, "Clusters": c_count})
            
    pd.DataFrame(results).to_csv("results_final.csv")
    print("SIT-Director Experiment Complete.")

if __name__ == "__main__":
    train(2000)
