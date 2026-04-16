# Plan Document: Custom JAX Env & Algo

Following the **Layered Workflow** from [WORKFLOW.md](../../WORKFLOW.md), this plan gates development into verifiable stages.

## 🗺️ The Roadmap

### 🟢 Phase 1: Custom MJX Environment (Layer 1: Skeleton)
Goal: Create a custom `.xml` and a JAX-compatible environment class.
- [ ] **Step 1.1**: Design MuJoCo XML (e.g., a simple 2-DOF fetcher).
- [ ] **Step 1.2**: Implement `MjEnv` class inheriting from `mujoco_playground`.
- [ ] **Step 1.3**: **Shape Test**: Run a random policy at 1000 steps per second (SPS) to verify JAX compilation and output shapes.

### 🔵 Phase 2: Custom JAX Algorithm (Layer 1 & 2)
Goal: Implement a simple JAX RL algorithm (e.g., REINFORCE or simple PPO).
- [ ] **Step 2.1**: **Prompt 1 (Explain)**: Distill the chosen algorithm theory.
- [ ] **Step 2.2**: **Skeleton**: Implement Networks (Flax) and Loss functions (Jitted).
- [ ] **Step 2.3**: **Smoke Test**: Run 1k steps on `CartpoleBalance` (verified baseline env). 
  - *Criteria*: No NaNs, finite losses, gradients flowing.

### 🟡 Phase 3: Integrated Training (Layer 3)
Goal: Train the custom algo on the custom env.
- [ ] **Step 3.1**: Run 100k steps.
- [ ] **Step 3.2**: Compare SPS vs. `mujoco_playground` baselines.
- [ ] **Step 3.3**: Verify learning curves.

### 🔴 Phase 4: Documentation & Content (Tier 1-3)
- [ ] **Step 4.1**: Create `CUSTOM_ALGO_EXPLAINED.md` (Prompt 6).
- [ ] **Step 4.2**: (Optional) 3Blue1Brown-style explainer for the env/algo (Prompt 7).

---

## 🚦 Verification Gates

| Gate | Requirement | Tool |
|------|-------------|------|
| **Shape** | Obs/Action/Reward shapes match EXPECTED | `jax.shape` |
| **Speed** | SPS > 10,000 (standard MJX expectations) | `RLTracker.log_sps` |
| **Convergence** | Loss decreases, Reward increases on Cartpole | Tensorboard |

## 🛠 Future Research (Prompt 9 & 10)
*Reserved for modifications after the baseline is verified.*
