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

## 2. Goal Validation: Complete Environment Suite
The following environments are available in the repository and serve as our validation targets. Performance targets are based on the paper's A100 baseline.

### 2.1 Locomotion Validation
| Environment | Timesteps | Paper SPS | Target Return |
| :--- | :---: | :---: | :--- |
| **Go1JoystickFlat** | 200M | 417,451 | ~900+ |
| **BarkourJoystick** | 100M | 385,920 | ~35+ |
| **BerkeleyHumanoidFlat**| 150M | 120,145 | ~25+ |
| **G1Joystick** | 200M | 106,093 | ~15+ |
| **HumanoidRun** | 100M | 91,617 | ~900+ |
| **SpotJoystick** | 100M | 404,931 | ~37+ |

### 2.2 Manipulation Validation
| Environment | Timesteps | Paper SPS | Target Return |
| :--- | :---: | :---: | :--- |
| **PandaPickCube** | 20M | 140,386 | ~1000 |
| **PandaOpenCabinet** | 40M | 136,007 | ~1500+ |
| **LeapCubeReorient** | 240M | 76,354 | ~300 |
| **LeapCubeRotateZAxis** | 100M | 76,354 | ~20|
| **AlohaSinglePeg** | 250M | 121,119 | ~480+ |
| **PandaPickPixels** | 5M | 38,015 | 100% Success |

### 2.3 DM Control MJX Port
| Environment | Timesteps | Paper SPS | Target Return |
| :--- | :---: | :---: | :--- |
| **CartpoleBalance** | 60M | 718,626 | 1000 |
| **CheetahRun** | 60M | 435,162 | 1000 |
| **AcrobotSwingup** | 100M | 752,092 | 1000 |


Below is an **approximate numerical table** read from the **bold mean curves** in each subplot, using the **final reward at the right edge** of each chart.

Because this is extracted visually from a figure, treat these as **rough estimates**, not exact values.

| Task                  | PPO final reward | SAC final reward | Higher final reward |
| --------------------- | ---------------: | ---------------: | ------------------- |
| AcrobotSwingup        |             ~220 |              ~50 | PPO                 |
| AcrobotSwingupSparse  |              ~10 |               ~0 | PPO                 |
| BallInCup             |             ~560 |             ~970 | SAC                 |
| CartpoleBalance       |            ~1000 |            ~1000 | Tie                 |
| CartpoleBalanceSparse |            ~1000 |            ~1000 | Tie                 |
| CartpoleSwingup       |             ~680 |             ~860 | SAC                 |
| CartpoleSwingupSparse |             ~180 |             ~800 | SAC                 |
| CheetahRun            |             ~900 |             ~920 | SAC                 |
| FingerSpin            |             ~560 |             ~920 | SAC                 |
| FingerTurnEasy        |             ~970 |             ~950 | PPO                 |
| FingerTurnHard        |             ~960 |             ~820 | PPO                 |
| FishSwim              |             ~630 |             ~500 | PPO                 |
| HopperHop             |             ~140 |             ~150 | SAC                 |
| HopperStand           |             ~200 |             ~220 | SAC                 |
| HumanoidRun           |             ~160 |             ~230 | SAC                 |
| HumanoidStand         |             ~780 |             ~900 | SAC                 |
| HumanoidWalk          |             ~450 |             ~900 | SAC                 |
| PendulumSwingup       |              ~70 |              ~80 | SAC                 |
| PointMass             |             ~900 |             ~900 | Tie                 |
| ReacherEasy           |             ~980 |             ~960 | PPO                 |
| ReacherHard           |             ~980 |             ~960 | PPO                 |
| SwimmerSwimmer6       |             ~580 |             ~450 | PPO                 |
| WalkerRun             |             ~600 |             ~730 | SAC                 |
| WalkerStand           |             ~980 |             ~970 | PPO                 |
| WalkerWalk            |             ~970 |             ~970 | Tie                 |

Quick summary:

* **PPO higher final reward:** 8 tasks
* **SAC higher final reward:** 12 tasks
* **Ties:** 4 tasks

Reply with **CSV** and I’ll turn this into a downloadable CSV table.


## 3. Experiment Design: Humanoid Efficiency
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
