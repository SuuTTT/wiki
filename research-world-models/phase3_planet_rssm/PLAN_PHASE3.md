# Phase 3 Plan: PlaNet & Recurrent State Space Models (RSSM)

## Objective
Reimplement **Planning with Deep Dynamics Models (PlaNet)** and the foundational **Recurrent State Space Model (RSSM)**. 
Unlike Phase 1/2 which trained an Actor-Critic/Evolutionary policy, Phase 3 marks our entry into pure **Model-Predictive Control (MPC)** in latency space, where the agent plays the game *entirely* by imagining futures at runtime and choosing the best action sequence.

***

## 1. Core Concepts to Implement

### A. The Recurrent State Space Model (RSSM)
The problem with a pure RNN (like in Phase 1) is that it is entirely deterministic and struggles with stochastic environments. The problem with pure VAEs is they lack memory.
RSSM combines both:
- **Deterministic State ($h_t$):** Powered by a GRU/RNN. It remembers the past.
- **Stochastic State ($s_t$):** A Gaussian latent variable (like a VAE).
- **The Split:** 
  - **Prior:** Predicts $s_t$ using only the past: $p(s_t | h_t)$
  - **Posterior:** Encodes the *actual* image $x_t$ combined with the past: $q(s_t | h_t, x_t)$
  - *Goal during training:* Force the Prior to match the Posterior via KL Divergence.

### B. Latent Overshooting
To ensure our model doesn't just memorize 1-step frames, we need to train the Transition Model (the Prior) to predict multiple frames into the future without seeing the actual images.

### C. The Cross-Entropy Method (CEM) Planner
We will strictly evaluate without an Actor network.
1. The agent is at step $t$.
2. It generates $N$ random action sequences of length $H$ (Horizon).
3. It uses the RSSM Transition Model to "dream" the outcomes of all $N$ sequences.
4. An auxiliary Reward Predictor scores the final states.
5. The CEM algorithm takes the top $K$ best imaginary action sequences and refits its initial distribution (mean/variance).
6. After $I$ iterations, the agent executes the *first* action of the absolute best sequence.

***

## 2. Implementation Roadmap (CleanRL Style)

- **Step 1: Environment & Replay Buffer**
  - We need a robust replay buffer that stores entire episodes (trajectories of pixels and actions), not just single transitions, because we must sample contiguous sequences for recurrent training (BPTT).

- **Step 2: Architecture Construction**
  - Build the Convolutional Encoder & Decoder.
  - Build the `RSSM` class containing the deterministic GRU and the stochastic linear layers.
  - Build the standard `RewardPredictor` (Linear layer off the concat of $h_t$ and $s_t$).

- **Step 3: The Training Loop**
  - **Loss Function:** 
    - `Reconstruction_Loss` (MSE between predicted image and real image).
    - `Reward_Loss` (MSE between predicted reward and real reward).
    - `KL_Loss` (KL Divergence pushing the Prior to match the Posterior).

- **Step 4: The Evaluation (CEM Planner)**
  - Write a fast vectorized batched planner function utilizing `jax.vmap` or massive PyTorch parallelization to simulate thousands of rollouts per frame at runtime.

***

## 3. Reference Material
- **Paper:** *PlaNet: Learning Latent Dynamics for Planning from Pixels* ([1811.04551](https://arxiv.org/abs/1811.04551)). Downloaded to this directory.
- **Original Source:** Google Research (`planet/` folder in this directory). Written in older TensorFlow. We will translate this to a monolithic PyTorch `.py` file.