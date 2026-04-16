# Learning Path: Mastering MuJoCo with CleanRL

This guide outlines a structured progression for mastering continuous control using MuJoCo, following the [Best Practice Workflow](../WORKFLOW.md) of layered verification.

## Phase 1: The Foundation (Continuous Action Spaces)
**Goal:** Understand the transition from discrete (DQN) to continuous (PPO-Continuous) action spaces.

1.  **Environment:** `Pendulum-v1`
    *   **Challenge:** Simple gravity-based continuous control.
    *   **Target Algorithm:** `ppo_continuous_tutorial.py`
    *   **Key Concept:** Gaussian Policy (parameterizing $\mu$ and $\sigma$ instead of $Q(s, a)$).
    *   **Baseline:** Mean return > -250 (Solved).
2.  **Verification:**
    *   Implement `NormalizeObservation` and `NormalizeReward` wrappers.
    *   Verify the `RescaleAction` wrapper is correctly mapping $[-1, 1]$ to environment bounds.

## Phase 2: Locomotion Basics (MLP-based MuJoCo)
**Goal:** Train standard "Easy" MuJoCo gym environments.

1.  **Environments:** `Hopper-v4`, `HalfCheetah-v4`
2.  **Algorithms:**
    *   **PPO**: For stability and ease of tuning.
    *   **SAC (Soft Actor-Critic)**: To experience maximum entropy RL and off-policy efficiency.
3.  **Metrics Contract:**
    *   `losses/policy_loss`, `losses/value_loss`.
    *   `losses/alpha` (for SAC) — monitor entropy temperature.
    *   `charts/episodic_return` — look for steady growth vs. abrupt collapse.

## Phase 3: Research-Scale MuJoCo (Hierarchical & GNN)
**Goal:** Scale to high-dimensional or multi-agent robotics.

1.  **Environments:** `Ant-v4`, `Humanoid-v4`, `Pistonball` (PettingZoo MuJoCo-like).
2.  **Advanced Architectures:**
    *   **GNNs for Robotics:** Treating body parts/joints as nodes in a graph for morphological generalization.
    *   **Hierarchical RL (HRL):** Training a manager to pick locomotion "skills" and a worker to execute joint torques.
3.  **Suitability:**
    *   `Humanoid-v4` is ideal for **World Model** research (predicting complex physics transitions).
    *   `AntMaze` is the gold standard for **HRL** (sparse navigation + complex locomotion).

## Phase 4: Hardware Acceleration (IsaacGym)
**Goal:** Eliminate the CPU bottleneck.

1.  **Tooling:** Transition from `gymnasium` to `IsaacGym`.
2.  **Workflow:**
    *   Benchmark **SPS (Steps Per Second)**. On local hardware, you should see a 10x-100x increase vs. standard MuJoCo.
    *   Validate learning parity: Ensure the agent learns at the same "step-wise" efficiency even when parallelized across thousands of envs.

---

## MuJoCo Acceptance Criteria (CleanRL Baselines)

These baselines are derived from the [Open RL Benchmark](https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/MuJoCo-CleanRL-s-PPO--VmlldzoxODAwNjkw) for PPO on MuJoCo-v2 tasks.

| Environment | Algorithm | Checkpoint | Target Return |
| :--- | :--- | :--- | :--- |
| `Ant-v2` | PPO | 1M Steps | ~2500 |
| `HalfCheetah-v2` | PPO | 1M Steps | ~2100 |
| `Hopper-v2` | PPO | 400k Steps | ~2000 (stable to 1M) |
| `InvertedDoublePendulum-v2` | PPO | 400k Steps | ~5500 |
| `Reacher-v2` | PPO | 400k Steps | ~ -6 |
| `Swimmer-v2` | PPO | 1M Steps | ~90 |
| `Walker2d-v2` | PPO | 1M Steps | ~3000 |

### Why do MuJoCo returns exceed 1000?
If you are coming from **DeepMind Control Suite (dm_control)**, you might be used to rewards being normalized to a $[0, 1]$ range per step, resulting in a maximum episode return of exactly **1000** (for a 1000-step horizon). 

However, in the **OpenAI Gym/Gymnasium MuJoCo** environments:
1.  **Unbounded Reward Sums**: Rewards are not capped at 1.0 per step. They often consist of multiple components: `forward_reward` (velocity), `control_cost` (energy penalty), and `surprised_reward` (staying alive).
2.  **Velocity Bonus**: In tasks like `HalfCheetah` or `Ant`, the reward is primary based on how fast the robot moves forward. There is no theoretical ceiling; the faster the robot runs, the higher the return.
3.  **Survival Bonus**: Environments like `Humanoid` or `Hopper` provide a constant reward for every step the agent remains upright.
4.  **Episode Length**: While most tasks have a 1000-step limit, the raw accumulation of these multi-component rewards can easily scale into the thousands (e.g., 5000+ for `Humanoid` or `InvertedDoublePendulum`).

## Prototyping Checklist
- [ ] **Shape Test**: Verify MLP hidden layers match observation structure.
- [ ] **Smoke Test**: Run 1k steps to ensure MuJoCo renderer isn't crashing (use `MUJOCO_GL=egl`).
- [ ] **Normalization**: Always use `VecNormalize` for MuJoCo to stabilize gradients.
