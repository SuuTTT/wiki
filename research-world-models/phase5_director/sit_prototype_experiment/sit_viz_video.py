import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from torch.distributions import Categorical
import networkx as nx
import community as community_louvain
import os

# 4x4 Deterministic Map
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
MAP_ARRAY = np.array([list(row) for row in MAP])

def map_to_numeric(m):
    num = np.zeros(m.shape)
    num[m == 'S'] = 0.8
    num[m == 'F'] = 1.0
    num[m == 'H'] = 0.0
    num[m == 'G'] = 0.5
    return num

def render_original_map():
    plt.figure(figsize=(5, 5))
    plt.imshow(map_to_numeric(MAP_ARRAY), cmap='RdYlGn')
    for i in range(4):
        for j in range(4):
            plt.text(j, i, MAP_ARRAY[i, j], ha='center', va='center', color='black', fontsize=15)
    plt.title("Original Map: FrozenLake 4x4 (S=Start, H=Hole, G=Goal)")
    plt.savefig("original_map.png")
    plt.close()
    print("Original map rendered to original_map.png")

def generate_trajectories(episodes=10, steps_per_ep=20):
    env = gym.make("FrozenLake-v1", desc=MAP, is_slippery=False)
    history = []
    adj = np.zeros((16, 16))
    
    for _ in range(episodes):
        s, _ = env.reset()
        for _ in range(steps_per_ep):
            a = env.action_space.sample()
            ns, r, term, trunc, _ = env.step(a)
            adj[s, ns] += 1
            # Run Louvain on current graph
            G = nx.from_numpy_array(adj)
            clusters = {i: 0 for i in range(16)}
            if G.number_of_edges() > 0:
                clusters = community_louvain.best_partition(G)
            
            history.append({
                "state": s,
                "next_state": ns,
                "clusters": clusters.copy()
            })
            if term or trunc: break
            s = ns
    return history

def render_video(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        data = history[frame]
        state = data["state"]
        clusters = data["clusters"]
        
        # Left: Original Map State
        grid_orig = map_to_numeric(MAP_ARRAY)
        ax1.imshow(grid_orig, cmap='RdYlGn', alpha=0.6)
        r, c = divmod(state, 4)
        ax1.plot(c, r, 'ro', markersize=15, label="Agent")
        ax1.set_title(f"Original Map (Step {frame})")
        
        # Right: SIT Abstraction
        grid_sit = np.zeros((4, 4))
        for s, cl in clusters.items():
            rs, cs = divmod(s, 4)
            grid_sit[rs, cs] = cl
        
        ax2.imshow(grid_sit, cmap='tab20')
        ax2.plot(c, r, 'wo', markersize=10, label="Agent")
        ax2.set_title("SIT DISCOVERY (Macro-states)")
        
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200)
    ani.save("sit_discovery_evolution.gif", writer='pillow')
    plt.close()
    print("Video evolution saved to sit_discovery_evolution.gif")

if __name__ == "__main__":
    render_original_map()
    hist = generate_trajectories(15)
    render_video(hist)
