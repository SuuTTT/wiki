# RL Environments Progression Guide

This wiki maps out how we will graduate from classic, basic control problems to fully accelerated, highly complex hardware environments.

## 1. Classic Control (CartPole-v1)
**Why start here?**
Before venturing into complex physics engines or image-processing workflows, testing your RL algorithm on discrete, state-vector simulation is the standard sanity check. CartPole has a very small state representation (4 variables: position, velocity, angle, pole velocity), and only 2 actions. It solves instantly on a CPU, establishing a rapid feedback loop.

## 2. Atari (e.g. BreakoutNoFrameskip-v4 via EnvPool)
**Why progress here?**
Atari introduces visual inputs. Instead of a vector of 4 explicit physics values, the agent only receives RGB pixel arrays and must learn feature extraction using Convolutional Neural Networks (CNNs).
*   **Wrappers:** Atari requires extensive preprocessing such as frame stacking (averaging pixels over 4 consecutive frames so the agent can perceive velocity), reward clipping, and grayscale conversion.

## 3. Mujoco (e.g. HalfCheetah-v4)
**Why progress here?**
Continuous control. Unlike CartPole or Atari where you press a button (discrete action), controlling a robot leg involves specifying torques (continuous values). This requires transitioning from algorithms like DQN (which predict discrete Q-values) to Actor-Critic methods like PPO, DDPG, or SAC that parameterize continuous Gaussian distributions.

## 4. IsaacGym / IsaacSim
**Why progress here?**
Massive parallelization. Usually, environments run on the CPU, forming a bottleneck while the GPU waits for data. IsaacGym leverages the GPU directly to simulate thousands of environments simultaneously without transferring data back-and-forth across the CPU-GPU bridge. Learning this enables training complex agents in minutes instead of days.

## 5. Specialized Research Environments

### GNN + RL (Multi-Agent & Particle Environments)
**Suited for:** `PettingZoo` and `MPE (Particle)`
*   **Why?** GNNs handle variable-sized inputs and model multi-agent interactions as graph edges. They provide natural **permutation invariance**, allowing the agent's policy to stay consistent regardless of agent ordering in the observation vector.

### World Models (High-Dimensional Visual & Physics)
**Suited for:** `Atari` and `High-Dimensional MuJoCo (e.g., Humanoid-v4)`
*   **Why?** World Models excel at compressing high-dimensional pixels into latent state spaces. Learning the environment dynamics $P(s'|s,a)$ enables "imagination-based" training, which is significantly more sample-efficient than pure model-free RL for complex physics or visual tasks.

### Hierarchical RL (Sparse Rewards & Long Horizons)
**Suited for:** `Atari (Montezuma's Revenge)` and `Procgen (Heist/Maze)`
*   **Why?** These tasks suffer from extreme **sparse rewards**. HRL decomposes long horizons into a manager (selecting sub-goals like "find the key") and a worker (executing the movement), making the exploration problem much more manageable than searching in the raw action space.
