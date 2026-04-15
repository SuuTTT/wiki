# Phase 4: The Dreamer Series - Changelog

## [2026-04-14] Created Real End-to-End DreamerV3 Architecture (`06_dreamer_v3.py`)

### Modified/Added
- Replaced the mock/toy placeholder V3 script with a fully viable, authentic end-to-end mathematical implementation of the DreamerV3 model.
- **Symlog and Symexp Integrations**: True Symlog transforms squashing all observations and targets, coupled with inverse Symexp functions yielding stable expansions during Value prediction.
- **Two-Hot Encoding Strategy**: Shifted Value Regression explicitly to 255-bucket Two-Hot Categorical Classification.
- **Discrete RSSM**: Added a discrete recurrent component mapping a matrix of `[32 Categories, 32 Classes]` powered by Gumbel-Softmax Straight-Through (`st_sample`) proxies for backpropagation.
- **KL Divergence Balancing**: Formulated KL balancing logic (Alpha = 0.8) pulling the Discrete Prior distributions safely towards detached Posterior representations.
- **PyTorch Safe Lambda-Returns**: Structured the Actor-Critic imagination phase strictly with detach buffers alongside frozen-parameter contexts to navigate PyTorch's inplace graph modification errors reliably.
- **TensorBoard Integration**: Added SummaryWriters for real-time visualization of the Model Loss, Actor Return, Critic Divergence, and KL distributions directly into `benchmark/06_dreamer_v3/` directories.
- Run setup integrated safely with headless `nohup` protocols.

## [2026-04-15] Bugfixes across DreamerV1, V2, V3 Algorithms

### Critical Fixes in RL Math
- **Dynamics Offset Bug Fixed in World Model**: Discovered that across `04_dreamer_v1.py`, `05_dreamer_v2_atari.py`, and `06_dreamer_v3.py`, the `step_prior()` and `step_forward()` models were receiving `action[t]` instead of `action[t-1]` while learning observation `obs[t]`. This improperly trained the GRU to reconstruct images using actions that hadn't happened yet! Shifted the sequence iterator to use `prev_acti = batch_act[:, t]` updating the memory properly for dynamic consistency.
- **True Lambda Returns in Actor-Critic**: Identified a severe failure in Actor return mapping wherein `returns == R + 0.99*V(Current_State)` instead of iterating backwards through N-step estimations. Completely rewrote `compute_lambda_returns()` integrating true TD(Lambda) horizon rewards across V1, V2, and V3 frameworks.
- **Dimensionality Alignment**: Re-indexed Actor-Critic sequences correctly so that gradients flow successfully from `$TD(\lambda)_{t}$` back through the states `$h_{t}$` up towards the Actor's sampled continuous/discrete trajectories without triggering PyTorch's backend graph disconnections.

### DreamerV2 / V3 Path Derivative vs. REINFORCE Fixes
- **Discrete Actor Gradients**: In `05_dreamer_v2_atari.py`, shifted the discrete Actor from Gumbel-Softmax Straight-Through proxies to a proper **REINFORCE** estimator with baseline advantage centering. Continuous controls in V1/V3 safely use Path Derivative, but Discrete controls must use REINFORCE to prevent divergence and allow `Pong-v5` to solve rather than guessing random values indefinitely.
- **Training Epoch Expansion**: Discovered that scripts were incorrectly truncating around ~100k steps or doing toy 500-step training cycles. Rewrote the `06_dreamer_v3.py` training sequence to execute a physical environment data-collection loop alongside periodic gradient updates. Adjusted `total_steps` dynamically: **1,000,000 Steps** for continuous tasks (`04_dreamer_v1`, `06_dreamer_v3`) and **50,000,000 Frames** for Atari targets (`05_dreamer_v2`) mirroring the exact scale configurations of the original implementations.
