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

# 8x8 Sparse Map
MAP_8x8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFVF",
    "FFFHFFFG"
]

def make_env(size=8):
    if size == 4:
        return gym.make("FrozenLake-v1", desc=["SFFF", "FHFH", "FFFH", "HFFG"], is_slippery=False)
    else:
        return gym.make("FrozenLake-v1", desc=MAP_8x8, is_slippery=False)

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
            return len(set(partition.values()))
        return 1

class AC(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, n_out), nn.Softmax(-1))
        self.val = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 1))

def train(mode="hierarchical", size=8, episodes=2000, frozen_sit=False):
    run_name = f"scaling_{mode}_s{size}_frz{frozen_sit}_{int(time.time())}"
    writer = SummaryWriter(f"logs/{run_name}")
    print(f"Starting {mode} | size={size} | frozen={frozen_sit}...")
    
    env = make_env(size)
    n_states = size * size
    graph = GraphTracker(n_states)
    
    if mode == "hierarchical":
        # FIXED: Input dimension for 8x8 is 64 + 64 (state + goal) = 128
        # Previously was hardcoded to 16, causing the shape mismatch
        worker = AC(n_states + n_states, 4)
        manager = AC(n_states, n_states)
        m_opt = optim.Adam(manager.parameters(), 5e-4)
    else:
        worker = AC(n_states, 4)
        manager, m_opt = None, None
        
    w_opt = optim.Adam(worker.parameters(), 1e-3)
    recent_rewards = deque(maxlen=100)
    
    for ep in range(episodes):
        s, _ = env.reset(); done = False; tot_rew = 0; steps = 0
        w_hist, m_hist = [], []
        
        # Insight: Wait for graph/clusters to stabilize before Manager starts learning/guiding
        active_hier = True
        if frozen_sit and ep < 500: active_hier = False

        # Initial goal
        s_oh = torch.zeros(n_states); s_oh[s] = 1.0
        if mode == "hierarchical":
            g_probs = manager.net(s_oh)
            dist = Categorical(g_probs)
            target = dist.sample() if active_hier else torch.tensor(0)
            target_val = manager.val(s_oh)
        
        while not done and steps < 200:
            s_oh = torch.zeros(n_states); s_oh[s] = 1.0
            
            if mode == "hierarchical":
                g_oh = torch.zeros(n_states); g_oh[target] = 1.0
                w_in = torch.cat([s_oh, g_oh])
                w_probs, w_val = worker.net(w_in), worker.val(w_in)
            else:
                w_in = s_oh
                w_probs, w_val = worker.net(w_in), worker.val(w_in)
            
            # Action selection
            eps = max(0.1, 0.9 - ep/1500)
            if np.random.rand() < eps: action = torch.tensor(np.random.randint(4))
            else: action = Categorical(w_probs).sample()
            
            ns, r, term, trunc, _ = env.step(action.item())
            done = term or trunc; steps += 1
            graph.update_edge(s, ns)
            
            if mode == "hierarchical":
                ns_cl = graph.clusters.get(ns, 0); curr_cl = graph.clusters.get(s, 0)
                # Intrinsic reward for reaching the target cluster
                ir = 0.2 if (active_hier and ns_cl == target.item() and curr_cl != target.item()) else -0.01
                w_hist.append((w_val, Categorical(w_probs).log_prob(action), ir + r))
                
                # Manager step if cluster changed or intrinsic goal hit or episode done
                if ns_cl != curr_cl or ir > 0.1 or done:
                    if active_hier:
                        r_m = float(r) * 10.0 + (1.0 if ir > 0.1 else 0)
                        m_hist.append((target_val, Categorical(g_probs).log_prob(target), r_m))
                    
                    # Update for next segment
                    ns_oh = torch.zeros(n_states); ns_oh[ns] = 1.0
                    g_probs = manager.net(ns_oh)
                    target_val = manager.val(ns_oh)
                    dist = Categorical(g_probs)
                    target = dist.sample() if active_hier else torch.tensor(0)
            else:
                w_hist.append((w_val, Categorical(w_probs).log_prob(action), float(r)))
            
            s = ns; tot_rew += r

        # Optimize
        for hist, opt in [(w_hist, w_opt), (m_hist, m_opt)]:
            if not hist or not opt: continue
            vals, log_probs, rews = zip(*hist)
            vals = torch.stack(vals).view(-1)
            R = []; curr_r = 0
            for r_val in reversed(rews):
                curr_r = r_val + 0.99 * curr_r
                R.insert(0, curr_r)
            R = torch.tensor(R, dtype=torch.float32)
            adv = R - vals.detach()
            loss = -(torch.stack(log_probs) * adv).mean() + 0.1 * F.mse_loss(vals, R)
            opt.zero_grad(); loss.backward(); opt.step()

        if ep % 20 == 0: graph.run_sit_optimization()
        recent_rewards.append(tot_rew)
        avg_rew = np.mean(recent_rewards)
        writer.add_scalar("charts/episodic_return", avg_rew, ep)
        if ep % 100 == 0: print(f"  {mode} Ep {ep}: AvgRew {avg_rew:.2f} | Frz={frozen_sit}")

    writer.close()
    print(f"  {mode} Complete | Frz={frozen_sit} | Final AvgRew: {avg_rew:.2f}")

if __name__ == "__main__":
    train(mode="hierarchical", size=8, episodes=2000, frozen_sit=True)
