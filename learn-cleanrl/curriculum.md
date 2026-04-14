# CleanRL Learning Curriculum

This curriculum guides you through the fundamental and advanced concepts of Deep Reinforcement Learning, logically structured to build your knowledge incrementally using the CleanRL repository.

## Part 1: Value-Based Methods

* **Deep Q-Network (DQN)** (`dqn_tutorial.py`)
  * *Concept:* DQN is the foundational algorithm that combines classic Q-learning with deep neural networks to approximate action values.
  * *Placement:* Placed first because it introduces the core mechanics of value-based reinforcement learning, experience replay, and target networks.
* **Categorical DQN (C51)** (`c51_tutorial.py`)
  * *Concept:* C51 learns a categorical distribution of expected returns across multiple bins rather than just predicting a single scalar expected value.
  * *Placement:* Placed after DQN to demonstrate how shifting from expected values to distributional reinforcement learning drastically improves stability and performance.
* **Rainbow DQN** (`rainbow_tutorial.py`)
  * *Concept:* Rainbow integrates several independent improvements to DQN (including distributional RL, multi-step returns, and prioritized replay) into a single algorithm.
  * *Placement:* Placed here as the ultimate culmination of value-based methods, combining previous isolated lessons into one high-performing agent.

## Part 2: Advanced Policy Gradients & Continuous Control

* **Proximal Policy Optimization (PPO)** (`ppo_tutorial.py`)
  * *Concept:* PPO is a highly stable on-policy actor-critic algorithm that uses a clipped surrogate objective to prevent destructively large policy updates.
  * *Placement:* Placed as the foundational introduction to policy gradient methods due to its industry-standard reliability and ease of tuning.
* **PPO Continuous Control** (`ppo_continuous_tutorial.py`)
  * *Concept:* Adapts the standard PPO architecture to output means and standard deviations for controlling environments with continuous action spaces.
  * *Placement:* Immediately follows standard PPO to show how discrete logic translates into sampling from continuous Gaussian distributions.
* **Deep Deterministic Policy Gradient (DDPG)** (`ddpg_tutorial.py`)
  * *Concept:* DDPG bridges DQN and policy gradients by using an off-policy actor-critic setup that outputs deterministic actions for continuous control.
  * *Placement:* Follows continuous PPO to contrast on-policy stochastic continuous control with off-policy deterministic control and replay buffers.
* **Soft Actor-Critic (SAC)** (`sac_tutorial.py`)
  * *Concept:* SAC is an off-policy continuous control algorithm that maximizes a trade-off between expected return and action entropy to encourage robust exploration.
  * *Placement:* Placed after DDPG to showcase the modern state-of-the-art in sample-efficient continuous control and maximum entropy frameworks.
* **PPO with Recurrence (PPO-LSTM)** (`ppo_lstm_tutorial.py`)
  * *Concept:* Injects a Long Short-Term Memory (LSTM) network into PPO to handle partially observable environments by persisting hidden states across steps.
  * *Placement:* Introduced after mastering basic continuous control to add the complexity of sequential memory and backpropagation through time.
* **Phasic Policy Gradient (PPG)** (`ppg_tutorial.py`)
  * *Concept:* PPG separates the training of the policy and the value function into distinct phases while sharing representations to boost sample efficiency.
  * *Placement:* Placed at the end of the policy gradient section to explore advanced architectural decoupling and auxiliary training phases.

## Part 3: Multi-Agent & High-Performance Environments

* **PPO Self-Play** (`ppo_selfplay_tutorial.py`)
  * *Concept:* Trains an agent by pitting it against historical snapshots of itself in a zero-sum, multi-agent competitive environment.
  * *Placement:* Placed here to expand the training paradigm from single-agent mastery to multi-agent reinforcement learning (MARL) dynamics.
* **PPO with EnvPool** (`ppo_envpool_tutorial.py`)
  * *Concept:* Leverages highly optimized, C++ based environment vectorization (EnvPool) to drastically parallelize and accelerate training throughput.
  * *Placement:* Serves as the capstone of the current curriculum to focus entirely on high-performance training systems and infrastructure engineering.

---

## Part 4: Future Curriculum (Advanced CleanRL Topics)

Based on what you've conquered so far (Values, Advanced Policy Gradients, Multi-Agent, and High-Performance environments), we have reached the cutting edge of modern RL theory. However, the CleanRL repository still has a few critically important concepts left to explore natively:

1. **Maximum Entropy Continuous Control (`sac_continuous_action.py` / `td3_continuous_action.py`)** 
   While we touched on continuous spaces with PPO, the dominant algorithms across industry robotics are **Soft Actor-Critic (SAC)** and **Twin Delayed DDPG (TD3)**. They handle off-policy replay buffer limits while balancing entropy differently.
2. **Exploration in Sparse Environments (`ppo_rnd_envpool.py`)**
   **Random Network Distillation (RND)** is an intrinsic motivation algorithm for games with notoriously sparse rewards (like Montezuma's Revenge), creating curiosity by predicting the output of an untrained, frozen random network.
3. **Hardware Acceleration via JAX & XLA (`ppo_atari_envpool_xla_jax.py`)**
   You've mastered PyTorch architecture locally, but CleanRL heavily supports migrating to **JAX / Flax** for XLA compiler vectorization across TPUs, scaling speeds to stratospheric levels.
4. **Robust Policy Optimization (`rpo_continuous_action.py`)** 
   A relatively new regularization technique that forces PPO to randomly sample from a uniform distribution (with a decaying epsilon) over the continuous action bounds to drastically smooth catastrophic drops in PPO's monotonic improvements.

---

## Part 5: Representation Learning for RL

Moving beyond direct end-to-end learning, these concepts focus on extracting rich, task-agnostic representations to dramatically improve sample efficiency, generalization, and structural understanding in RL.

* **CURL & RAD (Contrastive Learning and Data Augmentation)**
  * *Concept:* CURL leverages contrastive unsupervised learning on raw pixels to extract stable high-level features, while RAD proves that simple data augmentations (like random crop or translation) can independently achieve state-of-the-art sample efficiency.
  * *Placement:* Essential for understanding how modern computer vision techniques bridged the sample-efficiency gap between pixel-based and state-based RL.
* **Graph Neural Networks in RL (GNN+RL)**
  * *Concept:* Seminal works like "Relational Deep Reinforcement Learning" utilize GNNs and self-attention to model relations between dynamic entities in an environment. 
  * *Placement:* Critical for structured, combinatorial, or multi-object tasks (like StarCraft or physical assembly) where standard MLPs and CNNs fail to capture relational dynamics.
* **World Models & Dreamer**
  * *Concept:* Seminal papers by Ha & Schmidhuber (World Models) and Hafner et al. (Dreamer) cleanly separate environment representation (learned via VAEs/RNNs) from the controller, allowing the policy to train entirely within its own latent simulation.
  * *Placement:* The pinnacle of model-based representation learning, showcasing how latent planning yields massive leaps in long-horizon tasks and sample efficiency.

---

## Part 6: Accelerating RL with Next-Gen Frameworks

Pushing the boundaries of simulation speed and training throughput requires compiler optimizations, zero-copy memory transfers, and modern RL abstractions.

* **LeanRL & PyTorch 2.0**
  * *Concept:* A framework leveraging PyTorch 2.0's `torch.compile()`, `vmap`, and native operator optimizations to squeeze massive speedups out of standard RL algorithms without rewriting them in lower-level languages.
  * *Placement:* Demonstrates how to achieve compiled-language speeds while maintaining highly readable, Pythonic PyTorch codebases.
* **TorchRL**
  * *Concept:* DeepMind and Meta's highly modular PyTorch-native library offering optimized primitive components (like TensorDicts and specialized batched replay buffers) for composable RL.
  * *Placement:* Teaches the modern paradigm of building production-grade, decoupled RL systems using structured tensors.
* **MuJoCo Playground & Brax (JAX/XLA)**
  * *Concept:* Fully hardware-accelerated physics engines and vectorized environments that execute entirely on the GPU/TPU accelerator, side-stepping the CPU-GPU transfer bottleneck common in standard RL.
  * *Placement:* The cutting edge of continuous control; essential for scaling robotic RL training times from days to mere minutes by simulating thousands of complex robotic bodies in parallel.
* **NVIDIA Isaac Lab (formerly Isaac Gym)**
  * *Concept:* End-to-end GPU-accelerated massive parallel physics simulation designed specifically for reinforcement learning and sim2real transfer.
  * *Placement:* Vital for modern industry robotics, teaching how tens of thousands of environment instances can run simultaneously on a single GPU.