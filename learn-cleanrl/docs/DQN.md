# Deep Q-Network (DQN)

Welcome, beginner, to Deep Q-Networks.
DQN bridges the gap between classic Q-Learning (tabular reinforcement learning) and deep neural networks, acting as a baseline for deep discrete control.

## 1. The Bellman Equation and Q-Learning
The ultimate goal of RL is to maximize the expected cumulative sum of discounted rewards:
$$ R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k} $$

The **Action-Value Function**, $Q(s, a)$, predicts this return from a given state $s$ when taking a specific action $a$, and following the optimal policy thereafter.
$$ Q^*(s, a) = \mathbb{E} [r_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') | s_t = s, a_t = a] $$

We minimize the Temporal Difference (TD) error using Mean Squared Error (MSE):
$$ L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right] $$

Where:
*   $\theta$: The current neural network weights.
*   $\theta^-$: The target network weights.

## 2. Replay Buffers
Deep learning assumes independent and identically distributed (i.i.d.) data. But in RL, sequential transitions $(s, a, r, s')$ are highly correlated.
To fix this, we store transitions in a cyclic array (Replay Buffer) capable of holding up to `buffer_size` samples (e.g., $N = 10,000$). Instead of training on the final immediate transition, we sample randomly from this buffer, breaking the sequence correlation and avoiding catastrophic forgetting or learning loops.

## 3. Target Networks
If we optimize $Q(s, a; \theta)$ toward a moving target $\max_{a'} Q(s', a'; \theta)$ parameterised by the exact same network simultaneously, training becomes highly unstable (like a dog chasing its tail).
Instead, DQN duplicates the Q-network onto a frozen **Target Network** ($\theta^-$). The target network provides stable Q-value estimations ($TD_{target}$) and its parameters are updated only periodically (every `target_network_frequency` steps).
