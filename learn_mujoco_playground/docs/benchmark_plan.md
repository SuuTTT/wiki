# MuJoCo Playground Benchmark Strategy (MJX)

## 1. Paper Benchmark Reference
Based on the MuJoCo Playground (MJX) paper, we target high-throughput locomotion environments to demonstrate JAX's scalability.

| Environment | Task Category | Paper Timesteps | Paper Speed (Steps/Sec) | Target Return |
| :--- | :--- | :--- | :--- | :--- |
| **CartpoleBalance** | Tutorial | 60M | 718,000 | ~995-1000 |
| **HumanoidRun** | Locomotion | 100M | 91,000 | ~900+ |
| **Ant** | Locomotion | 50M | 1.1M | ~1000+ |
| **BarkourJoystick**| Locomotion | 200M | 385,000 | ~450+ (Avg Vel) |
| **BerkeleyHumanoid**| Locomotion | 100M | 120,000 | ~900+ |
| **G1Joystick** | Humanoid | 400M | 106,000 | ~900+ |
| **PandaPickCube** | Manipulation | 20M+ | 140,000 | ~450-500 |
| **PandaOpenCabinet**| Manipulation | 40M+ | 136,000 | ~600+ |
| **PandaPickPixels** | Vision | 5M | 38,000 | 100% Success |

## 2. Experiment Design: Humanoid Efficiency
For our benchmark, we will run `HumanoidRun` with optimized parameters for the RTX 3090.

### Configuration (Humanoid)
- **Total Timesteps:** 100,000,000 (100M)
- **Parallel Envs:** 4,096
- **Batch Size:** 1,024 (Matches Berkeley/G1 paper specs)
- **Unroll Length:** 32
- **Discounting:** 0.98
- **Expected Time:** ~10-15 minutes
- **Target Performance:** Return > 900 (Stable running gait)

### Configuration (Panda - Manipulation)
- **Total Timesteps:** 50,000,000 (50M)
- **Parallel Envs:** 2,048 (Higher density for manipulation)
- **Batch Size:** 512
- **Unroll Length:** 10
- **Expected Time:** ~15-20 minutes
- **Target Performance:** Return > 450 (Reliable cube picking)

### Expected Performance Calculation
Using the 3090, we expect:
- **VRAM:** ~4GB - 8GB (much higher than Cartpole due to Humanoid geometry/joints)
- **GPU Util:** 100%
- **Wall Clock:** ~10-15 minutes for 100M steps.

## 3. Reproduction Command
```bash
cd /workspace/wiki/learn_mujoco_playground/repo
export MUJOCO_GL=egl
export WANDB_API_KEY=...
nohup uv --no-config run train-jax-ppo \
    --env_name HumanoidRun \
    --num_timesteps 100000000 \
    --num_envs 4096 \
    --batch_size 2048 \
    --num_minibatches 32 \
    --use_wandb \
    --nolog_training_metrics \
    --num_evals 20 > /workspace/wiki/learn_mujoco_playground/logs/benchmark_humanoid.log 2>&1 &
```
