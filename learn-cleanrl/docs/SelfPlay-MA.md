# Multi-Agent Self-Play (MA)

In the CleanRL file `ppo_pettingzoo_ma_atari.py`, the **MA** stands for **Multi-Agent**.

Instead of interacting with an environment alone (like CartPole or Breakout), the agent plays inside a competitive or cooperative game alongside other entities (like Pong, where there is a Left Paddle and a Right Paddle). 

## 1. Parameter Sharing (Self-Play)
In Multi-Agent Reinforcement Learning, you could technically create two entirely separate neural networks:
* Network A: Learns to play the Left Paddle.
* Network B: Learns to play the Right Paddle.

However, researchers discovered this is computationally wasteful and wildly unstable (since both networks are constantly changing, the environment effectively becomes non-stationary). 

The modern solution is **Parameter Sharing**, which naturally creates **Self-Play**:
We instantiate *only one single Neural Network*. 
During gameplay, we ask that exact same Network to look at the board from the perspective of Player 1 and make an action, and then look at the board from the perspective of Player 2 and make an action. 

Why is this incredible?
1. **Double Data**: The single network is gathering TWICE the amount of data per frame, because it is controlling both sides.
2. **Automatic Curriculum**: As the network figures out a new trick to score on the right side, the left side of the network instantly "knows" that trick and is violently forced to invent a counter-strategy. They pull each other up by their own bootstraps.

## 2. PettingZoo and SuperSuit
Historically, managing multi-agent data loops was a nightmare. 
* Gym (Gymnasium) assumes $1$ action in $\rightarrow 1$ reward out. 
* **PettingZoo** is the Multi-Agent equivalent of Gym, properly managing independent agent states, rewards, and deaths.

To get PettingZoo games to plug cleanly into our standard Single-Agent PPO architecture, we use a wrapper library called **SuperSuit**.

### The Flattening Trick
If you ask for 8 parallel copies of `pong_v3` (which has 2 players per game), you actually have 16 living agents.
SuperSuit uses `ss.pettingzoo_env_to_vec_env_v1()`, which mathematically flattens those 16 agents into what looks exactly like a standard 16-environment single-agent array.
Our PPO code loops cleanly over `batch_size = 16 * 128 steps`. The PPO network has no idea it is playing a 2-player game against itself; it just thinks it's playing 16 independent games. The parameter sharing completely occurs structurally under the hood!