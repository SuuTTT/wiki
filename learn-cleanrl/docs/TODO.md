# TODO: Next-Generation RL Architecture (Abstraction, Hierarchy, & Planning)

This roadmap outlines the specific technical steps to build our new tutorial pathway focusing on Hierarchical Reinforcement Learning (HRL), Latent Skill Discovery, Task and Motion Planning (TAMP), and World Models. 

Each component will have a dedicated `<algorithm>_tutorial.py` implementing the theoretical models.

## Phase 1: Temporal Abstraction (HRL)
- [ ] **Implement The Options Framework (`option_critic_tutorial.py`)**
  - Read: *The Option-Critic Architecture (Bacon et al. 2017)*
  - Goal: Extend PPO to include a higher-level network that selects temporally extended "options" and a lower-level network that executes atomic actions until a termination function `b(s)` fires.
- [ ] **Implement FeUdal / Goal-Conditioned RL (`hiro_tutorial.py` or `uvfa_tutorial.py`)**
  - Read: *Data-Efficient Hierarchical Reinforcement Learning (HIRO - Nachum et al. 2018)*
  - Goal: Build a "Manager" network that outputs a latent goal vector every $c$ steps, and a "Worker" network that receives intrinsic reward for matching the transition step to the Manager's latent vector.

## Phase 2: Unsupervised Skill Discovery (Latent Space)
- [ ] **Implement DIAYN (`diayn_tutorial.py`)**
  - Read: *Diversity is All You Need (Eysenbach et al. 2018)*
  - Goal: Train an agent on an environment *without any extrinsic reward*. Maximize mutual information between the generated trajectory and a randomly sampled Gaussian/Categorical latent skill $z$.
  - Output: A library of pre-trained atomic skills (e.g., "walk forward", "do a backflip") bounded strictly by latent variable conditioning.

## Phase 3: Neuro-Symbolic Task & Motion Planning (TAMP)
- [ ] **Implement Object-Centric RL (`gnn_ppo_tutorial.py`)**
  - Read: *Relational Deep Reinforcement Learning (Zambaldi et al. 2018)*
  - Goal: Swap our standard MLPs for Graph Neural Networks (GNNs) and Self-Attention (Transformers). Treat the environment state not as a flat vector, but as a dynamic graph of nodes (objects) and edges (relations).
- [ ] **Implement Bi-Level Planning (`nsrt_tamp_tutorial.py`)**
  - Read: *Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning (Silver et al.)*
  - Goal: Integrate a discrete symbolic planner (like Fast Downward / A* running PDDL) at the high level, invoking continuous PPO/SAC controllers at the low level parameters.

## Phase 4: World Models & Imagination
- [ ] **Implement Dreamer-V1 (Core Theory) (`dreamer_tutorial.py`)**
  - Read: *Dream to Control/World Models (Ha & Schmidhuber, Hafner et al.)*
  - Goal: Separate the environment model from the agent. 
    1. Learn a Variational Autoencoder (VAE) to encode pixels to latents.
    2. Learn an Recurrent State Space Model (RSSM) to predict the next latent given an action.
    3. Train PPO entirely inside the latent "dream" without ever stepping the real environment.
