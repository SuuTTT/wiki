# Hierarchical Latent World Models

This directory contains a specialized research path designed to bridge the gap from foundational Deep RL to state-of-the-art Hierarchical Latent Space Planning and World Models. We build everything from scratch, incrementally following the seminal literature to master learning abstraction and model-based continuous control.

## The Research Path

### Phase 1: Foundational Predictive Dynamics (World Models)
* **Status:** In Progress
* **Target:** `01_world_models_vae_rnn.py`
* **Concept:** Compress high-dimensional pixel observations into a compact latent vector ($z$) using a Variational Autoencoder (VAE). Then, model the probabilistic forward dynamics in this latent space given an action using a Mixture Density Network-RNN (MDN-RNN). 
* **Reference:** [Ha & Schmidhuber (2018) "World Models"](https://arxiv.org/abs/1803.10122)

### Phase 2: Latent Space Planning (PlaNet & RSSM)
* **Status:** Planned
* **Target:** `02_planet_rssm.py`
* **Concept:** Introduce the Recurrent State Space Model (RSSM) to capture both deterministic and stochastic elements of the environment state. Use the Cross-Entropy Method (CEM) for Model Predictive Control (MPC) entirely inside the "imagined" latent space.
* **Reference:** [Hafner et al. (2019) "Learning Latent Dynamics for Planning from Pixels"](https://arxiv.org/abs/1811.04551)

### Phase 3: Actor-Critic in Imagination (Dreamer)
* **Status:** Planned
* **Target:** `03_dreamer_actor_critic.py`
* **Concept:** Replace the computationally heavy CEM planning phase at timestep inference with a parametric Actor-Critic architecture that learns by backpropagating analytic gradients directly through the frozen, differentiable RSSM latent dynamics.
* **Reference:** [Hafner et al. (2020) "Dream to Control: Learning Behaviors by Latent Imagination"](https://arxiv.org/abs/1912.01603)

### Phase 4: Unsupervised Skill Discovery (DIAYN)
* **Status:** Planned
* **Target:** `04_diayn_latent_skills.py`
* **Concept:** Implement "Diversity Is All You Need" (DIAYN) on top of our latent state representations to organically discover a discrete set of reusable skills purely via intrinsic motivation and mutual information maximization, without extrinsic environment rewards.
* **Reference:** [Eysenbach et al. (2018) "Diversity is All You Need: Learning Skills without a Reward Function"](https://arxiv.org/abs/1802.06070)

### Phase 5: Hierarchical Latent Planning (Director / Manager-Worker)
* **Status:** Planned
* **Target:** `05_director_hierarchical.py`
* **Concept:** Combine World Models with Hierarchical RL. Implement a Manager-Worker architecture where the Manager operates at a slow temporal abstract timescale setting latent-space subgoals, and the Worker operates at a fast timescale outputting primitive actions to achieve them.
* **Reference:** [Hafner et al. (2022) "Deep Hierarchical Planning from Pixels"](https://arxiv.org/abs/2206.04114)
