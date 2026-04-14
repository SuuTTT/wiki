# Rainbow DQN

Rainbow is not a single new algorithm—it is the Voltron of Deep Q-Networks. In 2017, researchers at DeepMind combined six independent improvements to the original DQN algorithm into a single architecture, resulting in state-of-the-art performance across the Atari 2600 benchmark.

Here are the six components assembled into Rainbow:

## 1. Double Q-Learning (DDQN)
**The Problem:** Standard DQN overestimates Q-Values because it uses the maximum Q-value for the next state as the target. Since the estimates are noisy, taking the maximum often results in taking the maximum of the positive noise.
**The Fix:** Separate Action Selection and Action Evaluation.
* Selection: Use the **Primary Network** to find the best action ($argmax_a Q(s', a)$).
* Evaluation: Use the **Target Network** to estimate the value of that specific action ($Q_{target}(s', a_{chosen})$).

## 2. Prioritized Experience Replay (PER)
**The Problem:** Uniformly sampling from a Replay Buffer wastes time on transitions the agent has already mastered or transitions that provide zero learning signal.
**The Fix:** Sample transitions proportionally to their **TD-Error** (how "surprised" the network was by the outcome). Massive errors get re-sampled often; boring transitions get ignored. This requires organizing memory into a mathematically efficient **Sum Segment Tree**.
* **Importance Sampling:** Because we bias our sampling toward high-error transitions, we must multiply the loss gradients by "Importance Sampling Weights" to correct the bias, avoiding shattering the network parameters.

## 3. Dueling Networks
**The Problem:** In many states, the action you take doesn't matter (e.g., waiting for an elevator). Standard DQN forces the network to calculate the exact Q-Value for every single useless action.
**The Fix:** Split the network's final layer into two separate streams:
* **Value Stream $V(s)$**: Answers "How good is it to be in this state generally?"
* **Advantage Stream $A(s, a)$**: Answers "How much better is taking action A compared to the average action?"
* The Q-Values are then recombined: $Q = V + (A - \text{mean}(A))$.

## 4. Multi-Step Learning (N-Step Returns)
**The Problem:** Standard DQN looks just 1 step ahead ($r + \gamma \max Q(s_{t+1}, a)$). If the reward is 100 steps away, it takes thousands of episodes for the reward to trickle back to the start.
**The Fix:** Instead of 1 step, accumulate $n$ steps of rewards before bootstrapping!
$Q_{target} = r_t + \gamma r_{t+1} + \dots + \gamma^n \max Q(s_{t+n}, a)$

## 5. Distributional RL (C51)
**The Problem:** Predicting the "Expected Value" averages out the variance. If jumping over a pit has a 50% chance of +100 and a 50% chance of -100, the expected value is 0.
**The Fix:** Refer to `wiki/C51.md`. Chop the expected returns into 51/101 distinct atoms (bins) and output a probability mass function across those bins. Minimize Cross-Entropy rather than Mean Squared Error.

## 6. Noisy Nets
**The Problem:** Epsilon-greedy (randomly pressing buttons 10% of the time) is a terrible way to explore complex environments because it lacks spatial awareness and consistency over time.
**The Fix:** Throw away Epsilon-Greedy. Instead, inject parameterized Gaussian noise directly into the weights and biases of the linear layers (`NoisyLinear`). 
* The network learns *how much* noise it wants. Over time, as it becomes more confident in winning states, it automatically dials the noise down to 0, smoothly transitioning from Exploration to Exploitation based on the state itself rather than an arbitrary `global_step` counter.