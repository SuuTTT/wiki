# Phase 4 Plan: The Dreamer Series (V1, V2, V3)

## Objective
Reimplement the **Dreamer** algorithm lineage (Dream to Control, DreamerV2, DreamerV3) introduced by Danijar Hafner.
In Phase 3 (PlaNet), we successfully built the RSSM and used computation-heavy CEM to plan without a policy. In Phase 4, we will eliminate the CEM planner entirely. We will train an **Actor-Critic** architecture *completely inside the latent imagination* of the World Model. 

The goal is to translate these complex, multi-file generalized frameworks (TensorFlow/JAX) into strict, highly-readable, single-file PyTorch **CleanRL** scripts.

***

## 1. DreamerV1 (Dream to Control - 2019)
**Core Innovation: Solving the SPS Bottleneck**
*   **Architecture:** Reuses the exact RSSM from Phase 3 (Continuous Gaussian latent space).
*   **The Shift:** Delete CEM. Add an `Actor` (Policy) and a `Critic` (Value Network).
*   **The Magic:** The RSSM provides a fully differentiable physics simulator. We use the Actor to take actions inside the imagined (Prior) state space. Because the transition dynamics and rewards are differentiable, we propagate analytic gradients of the reward straight back through the imaginary time steps into the Actor weights!
*   **Implementation Target (`04_dreamer_v1.py`):**
    *   Build Actor and Value networks operating purely on the `(h_t, s_t)` latent concatenation.
    *   Implement **Lambda Returns ($TD(\lambda)$)** to balance bias-variance over the imagined rollout horizon ($H=15$).

## 2. DreamerV2 (Mastering Atari with Discrete World Models - 2020)
**Core Innovation: Categorical Latent States & KL Balancing**
*   **The Shift:** Continuous Gaussian latents struggle to model sharp categorical transitions (like picking up a key, or an enemy appearing). DreamerV2 replaces the Gaussian `$s_t$` with a **Discrete Categorical** distribution ($N$ categorical variables each with $M$ classes).
*   **The Magic:** Uses straight-through estimators for backprop. It also introduces **KL Balancing** (scaling the Prior toward the Posterior faster than pulling the Posterior to the Prior, stabilizing representations).
*   **Implementation Target (`05_dreamer_v2.py`):**
    *   Replace `Normal(mean, std)` with `OneHotCategorical`.
    *   Implement KL-balancing coefficients in the objective.

## 3. DreamerV3 (Mastering Diverse Domains through World Models - 2023)
**Core Innovation: Universal Scaling & Symlog Predictions**
*   **The Shift:** Fixed hyperparameters across *all* domains (Atari, Continuous Control, Minecraft).
*   **The Magic:** 
    *   **Symlog Predictions**: Instead of normalizing raw rewards or images linearly, it compresses large magnitudes logarithmically while preserving sign: `symlog(x) = sign(x) * ln(|x| + 1)`. This prevents exploding gradients on massive returns.
    *   **Two-hot Regression**: Discretizing continuous prediction targets to stabilize value learning.
*   **Implementation Target (`06_dreamer_v3.py`):**
    *   Apply `symlog` transformations to the target data.
    *   Convert continuous losses to categorical cross-entropy.

***

## 4. The CleanRL Translation Strategy
The official repositories (`dreamerv1_repo/`, `dreamerv2_repo/`, `dreamerv3_repo/`) are heavily abstracted to support multi-node scaling and massive parallel environments.
We will strip away the multi-file abstraction hierarchy:
1.  **Single-File Logic:** A top-down python script.
2.  **Explicit Shapes:** Hardcode the tensor reshapes for clarity rather than relying on dynamic flattening utility functions.
3.  **Direct TensorBoard Logging:** Instead of proprietary logging pipelines, map the `kl`, `critic_loss`, `actor_loss`, and `reward` directly into standard `SummaryWriter`.
4.  **Environment:** Target `envpool` continuous/discrete domains to allow 10x faster local testing over `Gymnasium`.