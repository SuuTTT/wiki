https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
algo	env_id	mean_reward	std_reward	n_timesteps	eval_timesteps	eval_episodes
ppo	CarRacing-v0	153.938	74.465	4M	179000	179

ppo_lstm	CarRacing-v0	862.549	97.342	4M	149588	156


# research

For a **baseline on Gymnasium CarRacing-v3**, I’d start with **PPO + CnnPolicy**, using the **RL Baselines3 Zoo CarRacing-v3 config** as your reference. The environment exposes a continuous-action mode by default, and RL Zoo already includes a tuned PPO setup specifically for **CarRacing-v3** with image preprocessing and training settings: frame skip 2, resize to 64×64, grayscale, frame stack 2, 8 envs, and 4M timesteps. That makes it the easiest baseline to reproduce and compare against later. ([健身房文档][1])

If you want the **strongest simple SB3-style baseline** rather than the most standard one, I’d try **RecurrentPPO / PPO-LSTM** next. In the RL Zoo benchmark on the older **CarRacing-v0**, plain PPO scored about **153.9 ± 74.5**, while **ppo_lstm** reached about **862.5 ± 97.3** under the same 4M-step benchmark format. That is a big enough gap that recurrence is worth testing early for CarRacing-like visual control. ([GitHub][2])

I would **not** pick **SAC as the first baseline** here, even though SB3 notes that SAC handles bounded continuous actions more naturally than PPO because SAC uses tanh squashing while PPO clips actions. In practice, the official RL Zoo support around CarRacing is much more plug-and-play for PPO, and RL Zoo’s own benchmarking pipeline explicitly skipped CarRacing for SAC because its params required extra external code. For a first baseline, reproducibility usually matters more than theoretical fit. ([Stable-Baselines3 Docs][3])

My practical recommendation:

* **1st baseline:** PPO + CnnPolicy, RL Zoo CarRacing-v3 config
* **2nd baseline:** RecurrentPPO / PPO-LSTM
* **Only after that:** SAC, if you specifically want an off-policy continuous-control comparison ([GitHub][4])

If you want, I can turn the current RL Zoo CarRacing-v3 PPO setup into a minimal training script for SB3.

[1]: https://gymnasium.farama.org/environments/box2d/car_racing/ "Car Racing - Gymnasium Documentation"
[2]: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md "rl-baselines3-zoo/benchmark.md at master · DLR-RM/rl-baselines3-zoo · GitHub"
[3]: https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/ "Stable Baselines3 Documentation"
[4]: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml "rl-baselines3-zoo/hyperparams/ppo.yml at master · DLR-RM/rl-baselines3-zoo · GitHub"


Not as-is.

From the **CleanRL docs**, the strongest general recommendation among their **continuous-control, vector-observation** baselines is **RPO**, not plain PPO: the docs explicitly say to prefer `rpo_continuous_action.py` over `ppo_continuous_action.py`, and report that RPO was equal or better in **57/61** tested environments and better than PPO in **93%** of tested environments. But that claim is for tasks with **continuous actions and low-level/vector observations**, such as dm_control, Gym classic control, and MuJoCo-style setups. ([docs.cleanrl.dev][1])

**CarRacing-v3 is different.** Its observation is a **96×96×3 RGB image**, and by default it uses a **continuous 3D action space**; it can also be switched to a **5-action discrete** version with `continuous=False`. ([健身房文档][2])

That mismatch matters because CleanRL’s `ppo_continuous_action.py`, `rpo_continuous_action.py`, and `sac_continuous_action.py` are documented as working with **Box observations of low-level features**, not pixel inputs. Their implementations also use simple MLPs on the flattened observation rather than a CNN image encoder. ([docs.cleanrl.dev][3])

So my recommendation is:

* If you want the **least modification**, use **`ppo_atari.py` + `CarRacing-v3(continuous=False)`**. `ppo_atari.py` already has a CNN for pixel input and explicitly supports only **discrete** actions. But you will still need **some** edits, because it includes Atari-specific wrappers like `NoopResetEnv`, `MaxAndSkipEnv`, `EpisodicLifeEnv`, reward clipping, grayscale, resize, and frame stacking. ([GitHub][4])
* If you want **true continuous control**, start from **`sac_continuous_action.py`** and modify it to use a **CNN encoder** plus image preprocessing wrappers. `sac_continuous_action.py` is the CleanRL script intended for continuous actions, but it is **not pixel-ready out of the box**. ([docs.cleanrl.dev][3])

So the practical answer is:

**Best with minimal pain:** `ppo_atari.py` on **discrete CarRacing**.
**Best CleanRL continuous starting point:** `sac_continuous_action.py`, but **yes, you need to modify the original code**.
**I would not choose `rpo_continuous_action.py` unchanged for CarRacing**, even though RPO is the stronger recommendation in the docs, because that recommendation does not cover pixel-based CarRacing. ([docs.cleanrl.dev][5])

If you want, I can sketch the exact code changes for either:

1. **PPO-Atari → discrete CarRacing**, or
2. **SAC-continuous → pixel CarRacing**.

[1]: https://docs.cleanrl.dev/rl-algorithms/ppo/ "Proximal Policy Gradient (PPO) - CleanRL"
[2]: https://gymnasium.farama.org/environments/box2d/car_racing/ "Car Racing - Gymnasium Documentation"
[3]: https://docs.cleanrl.dev/rl-algorithms/sac/ "Soft Actor-Critic (SAC) - CleanRL"
[4]: https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari.py "raw.githubusercontent.com"
[5]: https://docs.cleanrl.dev/rl-algorithms/rpo/ "Robust Policy Optimization (RPO) - CleanRL"
