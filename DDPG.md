# Deep Deterministic Policy Gradient (DDPG)

Welcome to DDPG! Now that you know about Continuous Control (read `wiki/ContinuousControl.md`), DDPG is the classic starting point.
It is an *off-policy* algorithm that combines the Replay Buffers and Target Networks of DQN but applies an Actor-Critic architecture that handles continuous spaces.

## 1. The Deterministic Actor
Unlike PPO, where the Actor outputs probabilities over a discrete set, DDPG's Actor computes $a = \mu(s)$. It outputs a single exact continuous action vector. This policy is completely deterministic!
To explore the environment, we must explicitly add random Gaussian noise to the Actor's output before feeding it into the environment:
$$ a_t = \text{clip}(\mu(s_t) + \epsilon, a_{min}, a_{max}) \text{ where } \epsilon \sim \mathcal{N}(0, \sigma) $$

## 2. The Critic's Role
In DQN, we passed a State into the network and got $N$ outputs (a Q-value for every discrete action).
In DDPG, the action is a vector of floats. So we pass **BOTH the State AND the Action** into the Critic Network to get a single Q-Value:
$$ Q(s, a): (\mathbb{R}^{state} \times \mathbb{R}^{action}) \rightarrow \mathbb{R}^1 $$

## 3. The Actor Loss (Policy Gradient)
How do we train the Actor when it's just a deterministic function outputting an action vector? We ask the Critic!
The Actor's goal is to output an action that maximizes the Q-Value the Critic predicts.
So, the Actor Loss is simply the negative Q-Value output by the Critic when fed the Actor's proposed continuous action:
$$ J(\theta^\mu) = - \frac{1}{N} \sum_{i} Q(s_i, \mu(s_i)) $$
We maximize $Q$ by minimizing $-Q$.

## 4. Soft Target Updates (Polyak Averaging)
In DQN, we updated the Target Network entirely every 500 steps (a "hard update").
DDPG introduced **Soft Updates**, which incrementally blend the Target Network parameters $\theta'$ with the current network parameters $\theta$ at every single training step using a tiny fraction $\tau$ (usually $0.005$):
$$ \theta_{target} = \tau \theta + (1 - \tau) \theta_{target} $$

By updating both the Actor and Critic Target Networks softly, we stabilize learning.