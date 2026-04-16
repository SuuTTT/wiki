# MuJoCo MJX vs. PyTorch (CleanRL) Performance Analysis

## 1. Key Performance Observations (RTX 3090)

### Throughput and Parallelization
- **5x Speedup:** By increasing `num_envs` from 16 to 4096, we achieved a significant wall-clock speedup. The GPU is now saturated at 100% utilization.
- **Compute bound:** The RTX 3090 is maxing out its power draw (242W) and compute cores when running 4096 parallel environments, while wall-clock time remains low.

### VRAM Efficiency (MJX)
- **Extreme Efficiency:** VRAM usage remains low (~1GB) even with 4,096 parallel environments.
- **Reasoning:** MJX stores environment states as contiguous arrays. Unlike PyTorch, it does not create separate Python objects for each instance. State vectors for simple environments (like Cartpole) are very small.
- **Static Compilation:** JAX compiles the training loop into optimized machine code (XLA), minimizing the memory overhead of the computation graph.

### Training Overhead
- **Logging Impact:** Enabling `--log_training_metrics` adds ~20% wall-clock overhead due to frequent Device-to-Host (GPU to CPU) data transfers and synchronization.
- **Recommendation:** Use a high number of evaluation steps (`--num_evals`) instead of continuous training metric logging for maximum throughput.

## 2. Environment Complexity and Resource Scaling

| Category | Benchmark Tasks | Steps/Sec (3090) | Target Performance |
| :--- | :--- | :--- | :--- |
| **Simple** | `CartpoleBalance`, `Acrobot` | 14M+ | Return > 990 |
| **Locomotion** | `HumanoidRun`, `Ant` | 1.5M - 8M | Return > 900 (Stab.)|
| **Manipulation**| `PandaPickCube`, `PandaOpenCabinet` | 400k - 1.2M | Return > 450 (Success) |

### Performance Observations (RTX 3090)
- **High Sensitivity to `num_envs`:** For simple tasks, performance scales linearly up to thousands of envs. For `HumanoidRun`, the peak efficiency is around 2048-4096 envs.
- **Return Thresholds:** In MuJoCo Playground, rewards are typically normalized. A return of **1000** usually signifies a perfectly solved task (e.g., Cartpole not moving for 1000 steps). Locomotion tasks (Ant/Humanoid) often target **900+** for stability. Manipulation tasks use sparse/shaped rewards where **500+** indicates consistent task completion.

### Manipulation (Panda/Franka) Specifics
- **Collision Checking:** Panda environments require significantly more collision checking between the gripper and objects (e.g., `PandaPickCube`), which lowers the raw Steps/Sec compared to simple point masses.
- **Degrees of Freedom:** The high DoF of the Panda arm (7 joints + gripper) increases the size of the Jacobian and mass matrices, utilizing more of the 3090's tensor cores but still maintaining high efficiency due to mjx's vectorized physics.

## 3. Theoretical vs. Actual Steps
- JAX/MJX counts **total transitions** across all parallel environments.
- While the "step count" might look higher in JAX (e.g., millions vs. thousands), the wall-clock time is much lower because these transitions happen simultaneously on the GPU.
- PyTorch (CleanRL) implementations typically update more frequently with fewer environments, leading to higher "sample efficiency" in step count but much lower "time efficiency" on modern GPUs.
