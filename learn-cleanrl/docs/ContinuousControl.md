# Continuous Control: From Discrete Actions to Real Physics

Until now, we have only worked with **Discrete** environments like `CartPole` or Atari pixel games. In discrete environments, the action space is a finite list.
- CartPole: `[0 (Left), 1 (Right)]`
- Breakout: `[0 (Noop), 1 (Fire), 2 (Right), 3 (Left)]`

You can just run all states through a Neural Network, output $N$ numbers, and pick the one with the highest value (like DQN does). Or output $N$ probabilities (like PPO does).

## 1. What is Continuous Control?
Real-world robotics, physics simulators (like Mujoco/PyBullet), and self-driving cars do not operate on buttons. When a robot leg moves, the API needs the **Torque** applied to multiple joints. 
The Action Space is no longer an integer, it is a vector of floats. 
- *Hopper-v4 (Mujoco)*: `[Thigh Torque, Leg Torque, Foot Torque]`
- Action bounds: `[-1.0 to 1.0]` for each joint.

There are **infinite** possible actions. You cannot ask a DQN to output a Q-value for every possible float between -1.0 and 1.0.

## 2. Introducing Deterministic Policy Gradient methods (DDPG)
To handle infinite action spaces, we separate the network into an **Actor** and a **Critic**:
1. The **Actor** doesn't output probabilities. It outputs the *exact continuous variables* to take (e.g. `[0.45, -0.12, 0.99]`). 
2. The **Critic** takes *both* the State AND the continuous Action out of the Actor, and evaluates how good that combined pair is.

## 3. How do we Explore?
In DQN, we explored using *Epsilon-Greedy* (randomly pick a discrete action).
In PPO, we explored using *Categorical Entropy* (outputting a flat probability curve).
In continuous space, we explore using **Additive Noise**. We ask the Actor for exactly what it wants to do, and then we add random Gaussian noise to the float before passing it to the environment. 
$$ a_t = \mu(s_t) + \mathcal{N}(0, \sigma) $$
