# World Models: Design & Architecture

This document outlines the architectural pipeline, design choices, and telemetry metrics used in our specialized implementation of Hierarchical Latent World Models (starting with `01_world_models_vae_rnn.py`).

## 1. Algorithm Pipeline

The architecture is composed of three distinct neural networks working together. Unlike the original 2018 paper, our pipeline trains them **jointly**:

1. **Vision Model (VAE - Variational Autoencoder):**
   * **Input:** Raw pixel frames (64x64x3 RGB) from the environment.
   * **Function:** Compresses the high-dimensional visual space into a low-dimensional stochastic latent vector $z_t$.
2. **Memory Model (MDN-RNN):**
   * **Input:** The current latent vector $z_t$ and the action taken $a_t$.
   * **Function:** Maintains a hidden state $h_t$ over time (using an LSTM), and outputs a Mixture Density Network (MDN) distribution predicting the *next* latent state $z_{t+1}$. It serves as the predictive "world model" or imagination.
3. **Controller (PPO Actor-Critic):**
   * **Input:** The concatenated representation of the current moment ($z_t$) and the historical context ($h_t$).
   * **Function:** Outputs a continuous action distribution (Actor) and an expected return value (Critic) to navigate the environment.

## 2. Design Choices & Mathematical Intuition

### The Original Method: Modeling "Dreams"
The original 2018 methodology (Ha & Schmidhuber) was strictly sequential and froze the networks at each phase. 
1. The **VAE (Vision)** was trained on thousands of random, offline frames to simply learn how to reconstruct visual pixels into a compact latent vector ($z$).
2. The **MDN-RNN (Memory)** was then trained to predict $z_{t+1}$ given the frozen $z_t$ and action. This explicitly mimics human "imagination" or a "world model" &mdash; the RNN learns the rules of physics (like gravity and momentum) entirely inside its latent space rather than relying on the heavy 2D simulation. 
3. The **Controller (CMA-ES)** was an evolution strategy to solve the task using *only* the compressed features. A key revelation was that the agent could be trained *entirely inside its own hallucinated RNN dreams*, never once touching the real environment, and successfully transfer back to reality!

### Why Our Joint Implementation Does The Same (But Faster)
Instead of waiting hours for offline random rollouts to map the whole environment, our implementation fuses the Vision, Memory, and Controller via **Joint PPO Backpropagation**. By hooking the VAE and RNN directly to the PPO Actor-Critic:
* Real-time gradient signals force the VAE to compress features that are *structurally relevant* to the immediate reward function, filtering out useless background noise much earlier.
* The RNN still maintains a perfect sequence of the physics (since it's bounded by the VAE's MSE reconstruction loss), allowing the agent to continuously learn temporal dynamics *as* it drives, mimicking modern end-to-end continuous architectures (like Dreamer variants).

## 3. Telemetry & Metrics Tracking

When monitoring TensorBoard (`tensorboard --logdir benchmark`), track these critical variables to ensure the architecture hasn't mathematically exploded:

### Losses Category
* **`losses/vae_recon_loss`**: Pixel-perfect Mean Squared Error (MSE) between the real frame and the VAE's uncompressed reconstruction. If this doesn't drop steadily, the agent is functionally blind.
* **`losses/vae_kld`**: KL Divergence measures how close the latent distribution $z$ is to a standard normal distribution. If this hits exactly 0.0, it triggered "Posterior Collapse" (the VAE learned to ignore the image entirely).
* **`losses/agent_value_loss`**: The Mean Squared Error indicating how poorly the Critic guessed the true episodic return (via GAE). Massive spikes early in training are expected as the agent discovers catastrophic failures (e.g. driving off a cliff) and recalibrates its expectations.
* **`losses/agent_policy_loss`**: The core PPO clipped surrogate objective loss measuring actor updates.
* **`losses/approx_kl` & `losses/clipfrac`**: Standard PPO Trust Region metrics outlining how heavily the policy shifted during the current epoch update.
* **`losses/agent_value`**: The pure numeric value of the Critic's expected return.

### Debug Category
* **`debug/agent_entropy`**: The Actor's continuous exploration gauge. It starts high (pure random actions) and decays slowly as the agent commits to winning strategies. Instant crashes to `0` mean the agent got stuck repeating a single action.
* **`debug/latent_z_std` & `debug/latent_z_mean_abs`**: Tracks the absolute mean and standard deviation of the VAE's latent variables. Massive continuous spikes signify the VAE is degenerating mathematically, breaking the RNN's ability to predict sequences.
* **`debug/rnn_hidden_mean_abs`**: Tracks the magnitude of the LSTM's internal hidden states $h_t$ to ensure the memory tensor isn't overflowing to infinity.

## 4. Changelog
* **v1.2**: Added manual environment array buffering to faithfully track and log episodic returns (`charts/episodic_return`) alongside complex native EnvPool outputs.
* **v1.1**: Replaced basic structural Actor dummy-loss with the fully functional Generalized Advantage Estimation (GAE) Continuous PPO equations.
* **v1.0**: Migrated original CMA-ES flat linear controller to Joint PPO Actor-Critic methodology via `envpool` parallelization. Integrated explicit logging metrics (KL Divergence, Reconstruction Loss, etc.).
