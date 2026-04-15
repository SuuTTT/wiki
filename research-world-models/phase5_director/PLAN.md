# Phase 5: Director Algorithm - CleanRL Reimplementation Plan

## Overview
Director (Hafner et al., 2022) introduces deep hierarchical planning from pixels using a learned world model. It splits the policy into a **Manager** (high-level) and a **Worker** (low-level). 

## Architecture & Modifications
To reimplement Director in a single-file, easy-to-read "CleanRL" style based on our current `phase4_dreamer` foundation (which has already established our World Model baseline):

### 1. Goal Autoencoder
- Director introduces a Goal Autoencoder to discretize and regularize the goal representations given by the manager.
- We will add a Goal Autoencoder architecture working alongside the RSSM. It encodes states into latent subgoals.

### 2. High-Level Manager Policy
- Operates at a slower timescale interval $K$.
- Predicts future goals via the Goal Autoencoder space.
- Trained to maximize extrinsic environment rewards plus an intrinsic exploration bonus.

### 3. Low-Level Worker Policy
- Operates every environment step.
- Conditioned on the Manager's proposed latent goal.
- Receives intrinsic reward for achieving the Manager's latent subgoals.
- We will update the Actor-Critic pipeline so the Worker uses these feature conditionings.

### 4. Single-File CleanRL Structure
- `01_director_v1.py`
  - Re-use the optimized `DiscreteRSSM` and `TwoHotEncoding` modules from DreamerV3.
  - Implement `Manager` and `Worker` specific MLPs.
  - Construct the hierarchical imagination rollout loop:
    1. Unroll World Model.
    2. Manager imagines goals.
    3. Worker acts towards goals.
    4. Compute $\text{TD}(\lambda)$ for both levels.
  - Employ `cleanrl_utils.logger.RLTracker` for plotting unified hierarchical metrics.
