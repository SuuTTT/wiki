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

# 4x4 Deterministic Map
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]

def make_env():
    return gym.make("FrozenLake-v1", desc=MAP, is_slippery=False)

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

class AC(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_out), nn.Softmax(-1))
        self.val = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 1))

def train(episodes=2000):
    run_name = f"sit_v3_run_{int(time.time())}"
    writer = SummaryWriter(f"logs/{run_name}")
    print(f"Logging to TensorBoard: {run_name}")
    
    env = make_env()
    graph = GraphTracker(16)
    worker = AC(16 + 8, 4)
    manager = AC(16, 8)
    w_opt = optim.Adam(worker.parameters(), 1e-3)
    m_opt = optim.Adam(manager.parameters(), 5e-4)
    
    start_time = time.time()
    global_step = 0
    recent_rewards = deque(maxlen=100)
    
    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        tot_rew = 0
        w_hist, m_hist = [], []
        
        s_oh = torch.eye(16)[s]
        g_probs, m_val = manager.net(s_oh), manager.val(s_oh)
        dist = Categorical(g_probs)
        target = dist.sample()
        
        steps = 0
        total_ir = 0
        while not done and steps < 100:
            g_oh = torch.eye(8)[target]
            w_in = torch.cat([s_oh, g_oh])
            w_probs, w_val = worker.net(w_in), worker.val(w_in)
            
            # Epsilon-greedy exploration for sparse reward
            epsilon = max(0.05, 0.5 - ep/1000)
            if np.random.rand() < epsilon:
                action = torch.tensor(np.random.randint(4))
            else:
                action = Categorical(w_probs).sample()
            
            ns, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            steps += 1
            global_step += 1
            
            graph.update_edge(s, ns)
            ns_cl = graph.clusters.get(ns, 0)
            curr_cl = graph.clusters.get(s, 0)
            
            # Intrinsic reward for reaching manager's goal cluster
            ir = 1.0 if ns_cl == target.item() and curr_cl != target.item() else -0.01
            total_ir += ir
            w_hist.append((w_val, Categorical(w_probs).log_prob(action), ir))
            
            if ns_cl != curr_cl or ir > 0 or done:
                # Manager receives sparse reward + progress bonus
                r_m = float(r) + (0.1 if ir > 0 else 0)
                m_hist.append((m_val, dist.log_prob(target), r_m))
                s_oh = torch.eye(16)[ns]
                g_probs, m_val = manager.net(s_oh), manager.val(s_oh)
                dist = Categorical(g_probs)
                target = dist.sample()
            else:
                s_oh = torch.eye(16)[ns]
            s = ns
            tot_rew += r

        # Optimize
        w_loss_val, m_loss_val = 0, 0
        for hist, opt in [(w_hist, w_opt), (m_hist, m_opt)]:
            if not hist: continue
            vals, logs, rews = zip(*hist)
            vals = torch.stack(vals).squeeze()
            if vals.dim() == 0: vals = vals.unsqueeze(0)
            R = []
            curr_r = 0
            for r in reversed(rews):
                curr_r = r + 0.99 * curr_r
                R.insert(0, curr_r)
            R = torch.tensor(R, dtype=torch.float32)
            adv = R - vals.detach()
            
            loss = -(torch.stack(logs) * adv).mean() + 0.5 * F.mse_loss(vals, R)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if opt == w_opt: w_loss_val = loss.item()
            else: m_loss_val = loss.item()

        # SIT Optimization every 10 eps
        if ep % 10 == 0:
            num_cl = graph.run_sit_optimization()
            writer.add_scalar("SIT/Clusters", num_cl, ep)

        recent_rewards.append(tot_rew)
        avg_rew = np.mean(recent_rewards)
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/extrinsic_reward", tot_rew, ep)
        writer.add_scalar("charts/episodic_return", avg_rew, ep)
        writer.add_scalar("charts/intrinsic_reward", total_ir, ep)
        writer.add_scalar("charts/SPS", sps, ep)
        writer.add_scalar("losses/worker_loss", w_loss_val, ep)
        writer.add_scalar("losses/manager_loss", m_loss_val, ep)
        
        if ep % 100 == 0:
            print(f"Ep {ep} | AvgRew: {avg_rew:.2f} | SPS: {sps} | SIT: {len(set(graph.clusters.values()))}")

    writer.close()
    return run_name

if __name__ == "__main__":
    run_id = train(2000)
    print(f"Done. Run ID: {run_id}")
