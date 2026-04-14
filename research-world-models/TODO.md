# Research TODOs

## Phase 1 (Completed)
- [x] Implement joint PPO + VAE + MDN-RNN for state representation learning (`01_world_models_vae_rnn.py`).
- [x] Build Pure RL Baseline (`02_ppo_baseline_cnn_rnn.py`) to verify representation learning advantage.
- [x] Configure TensorBoard logging and video validation scripts.
- [ ] Decode and visualize the learned latent space representations and dynamics predictions to verify learning quality.

## Phase 2: True World Models (David Ha's Dreaming)
- [ ] Git clone David Ha's original World Models repository (`https://github.com/hardmaru/WorldModelsExperiments`) as a reference.
- [ ] Reimplement Ha's original method in CleanRL style:
  - Collect random rollouts to pre-train VAE offline (MSE/KLD).
  - Train MDN-RNN offline to predict next latent states with categorical/Gaussian distributions.
  - Train the Controller (CMA-ES or PPO) strictly inside the "imagined" latent environment.
- [ ] Compare "dreaming" performance directly with the Phase 1 representation learning baseline.

## Phase 3: RSMM & PlaNet (In Progress)
- [x] Create directory `phase3_planet_rssm`.
- [x] Clone Google Research's `planet` repository.
- [x] Download the seminal paper ([1811.04551](https://arxiv.org/abs/1811.04551)).
- [x] Generate a structural implementation plan (`PLAN_PHASE3.md`).
- [ ] Implement the core Recurrent State Space Model (RSSM).
- [ ] Implement the Cross-Entropy Method (CEM) planner for Model Predictive Control without a policy network.

## Phase 4: Dreamer Series
- [ ] Implement DreamerV1 (Actor-Critic inside latent space derivatives).
- [ ] Implement DreamerV2 / V3 advanced representations.

## Phase 5: Hierarchical World Models
- [ ] Develop Director/Manager-Worker hierarchical planning within latent representations.