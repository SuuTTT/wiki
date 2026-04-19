import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import networkx as nx
import community as community_louvain
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

# --- Environment ---
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
def make_env():
    return gym.make("FrozenLake-v1", desc=MAP, is_slippery=False)

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

# --- Networks ---
class AC(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_out), nn.Softmax(-1))
        self.val = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 1))

# --- Main Training ---
def train(episodes=2000):
    writer = SummaryWriter("logs/sit_director_final")
    env = make_env()
    graph = GraphTracker(16)
    worker = AC(16 + 8, 4)
    manager = AC(16, 8)
    w_opt = optim.Adam(worker.parameters(), 1e-3)
    m_opt = optim.Adam(manager.parameters(), 5e-4)

    win_rate_window = []
    
    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        tot_extr_rew = 0
        tot_intr_rew = 0
        w_hist, m_hist = [], []
        
        s_oh = torch.eye(16)[s]
        g_probs, m_val = manager.net(s_oh), manager.val(s_oh)
        dist = Categorical(g_probs)
        target = dist.sample()
        
        steps = 0
        while not done and steps < 100:
            g_oh = torch.eye(8)[target]
            w_in = torch.cat([s_oh, g_oh])
            w_probs, w_val = worker.net(w_in), worker.val(w_in)
            action = Categorical(w_probs).sample()
            
            ns, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            steps += 1
            graph.update_edge(s, ns)
            ns_cl = graph.clusters.get(ns, 0)
            curr_cl = graph.clusters.get(s, 0)
            
            # Intrinsic reward
            ir = 1.0 if ns_cl == target.item() and curr_cl != target.item() else -0.01
            tot_intr_rew += ir
            w_hist.append((w_val, Categorical(w_probs).log_prob(action), ir))
            
            if ns_cl != curr_cl or ir > 0 or done:
                m_hist.append((m_val, dist.log_prob(target), float(r) + (0.1 if ir > 0 else 0)))
                s_oh = torch.eye(16)[ns]
                g_probs, m_val = manager.net(s_oh), manager.val(s_oh)
                dist = Categorical(g_probs)
                target = dist.sample()
            else:
                s_oh = torch.eye(16)[ns]
            s = ns
            tot_extr_rew += r

        # Optimization & Loss Logging
        w_loss = 0
        if w_hist:
            vals, logs, rews = zip(*w_hist)
            vals = torch.stack(vals).squeeze()
            if vals.dim() == 0: vals = vals.unsqueeze(0)
            R = []
            curr_r = 0
            for r in reversed(rews):
                curr_r = r + 0.95 * curr_r
                R.insert(0, curr_r)
            R = torch.tensor(R, dtype=torch.float32)
            adv = R - vals.detach()
            w_loss = (F.mse_loss(vals, R) - (torch.stack(logs) * adv).mean()).item()
            w_opt.zero_grad(); (F.mse_loss(vals, R) - (torch.stack(logs) * adv).mean()).backward(); w_opt.step()

        m_loss = 0
        if m_hist:
            vals, logs, rews = zip(*m_hist)
            vals = torch.stack(vals).squeeze()
            if vals.dim() == 0: vals = vals.unsqueeze(0)
            R = []
            curr_r = 0
            for r in reversed(rews):
                curr_r = r + 0.99 * curr_r
                R.insert(0, curr_r)
            R = torch.tensor(R, dtype=torch.float32)
            adv = R - vals.detach()
            m_loss = (F.mse_loss(vals, R) - (torch.stack(logs) * adv).mean()).item()
            m_opt.zero_grad(); (F.mse_loss(vals, R) - (torch.stack(logs) * adv).mean()).backward(); m_opt.step()

        # Metrics
        win_rate_window.append(1.0 if tot_extr_rew > 0 else 0.0)
        if len(win_rate_window) > 100: win_rate_window.pop(0)
        win_rate = np.mean(win_rate_window)

        # TensorBoard Logging
        writer.add_scalar("Reward/Extrinsic", tot_extr_rew, ep)
        writer.add_scalar("Reward/Intrinsic", tot_intr_rew, ep)
        writer.add_scalar("Reward/WinRate_100", win_rate, ep)
        writer.add_scalar("Loss/Worker", w_loss, ep)
        writer.add_scalar("Loss/Manager", m_loss, ep)

        if (ep + 1) % 100 == 0:
            c_count = graph.run_sit_optimization()
            writer.add_scalar("SIT/Clusters", c_count, ep)
            print(f"Ep {ep+1} | Extr: {tot_extr_rew} | Intr: {tot_intr_rew:.2f} | WinRate: {win_rate:.2f} | Clusters: {c_count}")

    writer.close()
    print("Final Experiment Complete. Logged to TensorBoard.")

if __name__ == "__main__":
    train(2500)
