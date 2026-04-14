# Recurrent Proximal Policy Optimization (PPO-LSTM)

Up until this point, every algorithm we have implemented (DQN, PPO, DDPG, SAC, Rainbow) relies on the **Markov Property**: The assumption that the current state $S_t$ contains *all* the information necessary to predict the future. 

But what if it doesn't?

## 1. Partially Observable Environments (POMDPs)
Imagine playing Atari Pong. If you take a single snapshot (1 frame) of the game, you can see where the ball is, but you cannot determine its *velocity* or *direction*. The environment is **"Partially Observable"**. 
* **Standard Fix:** "Frame Stacking". Stacking the last 4 frames together allows the network to infer velocity.
* **The Problem:** Frame stacking works for 4 frames of Pong, but what if you are navigating a maze and need to remember a key you saw 500 steps ago? Frame stacking 500 images is computationally impossible.

## 2. Enter Recurrent Neural Networks (LSTMs)
**Long Short-Term Memory (LSTM)** networks are designed specifically to carry hidden information across time steps. By injecting an LSTM layer between our feature extractor (CNN or Linear layers) and our Actor/Critic heads, we give the agent a "Memory".
* At step $t=0$, the LSTM receives the state $S_0$ and creates a hidden state $H_0$.
* At step $t=1$, the LSTM receives the state $S_1$ AND the hidden state $H_0$.
* At step $t=100$, the agent can execute actions based on something it saw at $t=0$, because $H_{99}$ has carried that information forward through time!

## 3. Backpropagation Through Time (BPTT) 
In standard PPO, we shuffle all the transitions in our rollout buffer completely randomly.
With PPO-LSTM, we **cannot** shuffle transitions randomly, because the LSTM relies on the chronological sequence of events.
1. We collect a rollout of exactly $128$ steps across $N$ parallel environments.
2. We save the `initial_lstm_state` from step 0.
3. During optimization, we pass the *entire unbroken chronological sequence* of 128 steps into the LSTM, starting from that initial state.
4. PyTorch automatically builds a massive computational graph over all 128 steps and calculates gradients backwards through time.

## 4. The "Done" Reset Mask
If an agent dies on step 14, step 15 is a brand new game. If we don't wipe the agent's memory, it will think the new game is a continuation of the old one, bleeding the trauma of losing into the fresh start.
To fix this, we pass a `done` mask into the LSTM logic. During the sequence loop, if `done == True`, we forcefully multiply the hidden state $H$ and cell state $C$ by `0.0`, wiping the agent's memory perfectly on the frame the new episode begins.