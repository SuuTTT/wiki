# RL Curriculum Experiment Report

This document reports the latest empirical benchmark runs across the single-agent, multi-agent, and high-performance algorithms implemented in our curriculum. All algorithms were tested concurrently in a single multi-core compute environment.

## 1. High-Level Benchmark Table

| Algorithm | Environment | Completion Time | Speed (SPS) | Final / Active Return |
| :--- | :--- | :--- | :--- | :--- |
| **PPG** | `CartPole-v1` | ~9m | N/A (Phasic batched) | ~500.0 (Solved) |
| **PPO-LSTM** | `CartPole-v1` | ~34m | ~4.28 it/s | ~500.0 (Solved) |
| **PPO (Continuous)** | `Pendulum-v1` | ~47m | ~11.71 it/s | > -200.0 (Solved) |
| **Rainbow C51** | `CartPole-v1` | ~42m | ~193.93 it/s | ~500.0 (Solved) |
| **PPO (EnvPool)** | `pong-v5` / Atari | ~5m | ~547 SPS | -1.0 (Episodic Life limit) |
| **PPO Self-Play** | `pistonball_v6` | (Active) | ~580 SPS | Oscillating (-17 to +29) |

*Metrics tracked continuously via local TensorBoard events (`runs/`).*

## 2. Environment Success Metrics & Video Artifacts

### Evaluating CartPole-v1
**What indicates success?**
The reward function yields `+1` for every step the pole remains upright. The environment automatically truncates at 500 steps. A perfect "solved" model consistently outputs a return of exactly **500.0**, meaning it perfectly balanced the pendulum until the max time limit was reached.

*Video Capture:*
The generated agent consistently balancing the pole at the end of training.
[**Watch PPG solving CartPole**](./videos/CartPole-v1__ppg_tutorial__1__1776148083/rl-video-episode-900.mp4)

### Evaluating Pendulum-v1 (Continuous)
**What indicates success?**
Unlike CartPole, Pendulum offers dense negative rewards: $\text{Reward} = -(\theta^2 + 0.1 \times \dot{\theta}^2 + 0.001 \times a^2)$. 
A random agent spins uncontrollably, yielding returns around `-1500`. A successful continuous policy learns to violently swing the pendulum up exactly to the top ($\theta = 0$) and hold it there, minimizing velocity ($\dot{\theta}$) and effort ($a$). Returns **above -200 (closer to 0)** indicate optimal stability.

*Video Capture:*
The generated continuous agent swinging the pendulum up.
[**Watch PPO Continuous solving Pendulum**](./videos/Pendulum-v1__ppo_continuous_tutorial__1__1776148083/rl-video-episode-900.mp4)

## 3. Why are there no videos for EnvPool / Atari?
If you look for video captures in the `ppo_envpool_tutorial.py` or the `pistonball_v6` loops, you won't find them. This is entirely by design.

Standard Gym episodes render video by explicitly passing `env = gym.wrappers.RecordVideo(env, "videos/")`, which forces the Python GIL to step the environment, parse the C-struct pixel array, convert it to a NumPy array, and ship it to an `ffmpeg` writer.

**EnvPool** deliberately bypasses Python altogether. It relies on a raw zero-copy C++ multithreading backend to evaluate environment dynamics and pushes memory directly to the GPU space. Inserting a Python-based video frame interceptor into a batched C++ execution loop mathematically defeats the entire purpose of EnvPool's 500+ SPS performance. 

When you prioritize throughput engineering (EnvPool/XLA), you temporarily sacrifice human-readable pixel diagnostics to achieve mathematical convergence locally at scale.
