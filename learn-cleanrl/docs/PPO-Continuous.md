# PPO for Continuous Actions

Until now, our Proximal Policy Optimization (PPO) networks outputted logits passed through a `Categorical` distribution (Softmax). If there were 4 discrete actions, the Actor output 4 values representing the probability of choosing each.

But what if you are controlling a robot arm? Your action isn't "Button A" or "Button B". Your action is $0.782$ Volts to motor 1, and $-0.443$ Volts to motor 2. This is a **Continuous Action Space**.

## 1. The Normal (Gaussian) Distribution
Instead of predicting probabilities across finite choices, the Actor neural network predicts the parameters of a **Gaussian Distribution** for every single motor joint:
1. **Mean ($\mu$)**: The expected value, where the network "thinks" the optimal action is. This is the output of the final Linear layer in the Actor.
2. **Standard Deviation ($\sigma$)**: How unsure the network is. If $\sigma$ is high, the agent explores widely. If $\sigma$ is low, it highly exploits the Mean.

In most Continuous PPO implementations (like CleanRL's), the `std` is **not** an output of the Neural Network based on the state. It is a separate, global, trainable PyTorch Parameter. The network only predicts the Mean.

## 2. Sampling and Log Probs
When the Actor predicts $\mu = 0.5$ and $\sigma = 0.2$ for a joint:
1. Under the hood, PyTorch creates `Normal(loc=0.5, scale=0.2)`.
2. The agent calls `.sample()` to draw a random number from the bell curve (e.g., $0.62$).
3. The agent calculates `.log_prob(0.62)` which calculates exactly how *probable* it was to pick that exact number from that specific curve. 
4. The environment executes the $0.62$ voltage command, returning a state and reward.

## 3. PPO Clipping in Continuous Space
The core PPO equation remains mathematically identical! 
1. We compute Advantage using GAE.
2. In the optimization loop, we recreate the bell curve using the *new* updated network weights predicting a new $\mu_{new}$.
3. We check the new probability of taking the historical action ($0.62$) and compare it: $\frac{P(0.62 | \mu_{new}, \sigma_{new})}{P(0.62 | \mu_{old}, \sigma_{old})}$.
4. If this ratio exceeds $1 + 0.2$, we clip the objective function just like discrete PPO.