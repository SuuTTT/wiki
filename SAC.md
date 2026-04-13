# Soft Actor-Critic (SAC)

If DDPG is the classic starting point for Continuous Control, **SAC** is the current state-of-the-art that you will see used in almost all real robotics applications.

## 1. The Core Problem with DDPG
1. **Overestimation Bias**: Because DDPG's Critic updates its value based on the **Maximum** Q-Value of the next state, it acts optimistically. It sees a random spike in noise as a "good action" and updates the actor to exploit it. The Actor and Critic spiral into believing terrible actions are optimal.
2. **Exploration vs Exploitation**: DDPG's exploration is "dumb"—it just adds Gaussian noise to the action. Over time, the policy converges instantly and completely stops exploring.

## 2. Twin Delayed Critic (To Fix Overestimation)
Instead of having 1 Critic network, SAC has **Two Critic Networks** (`qf1` and `qf2`).
When calculating the Target Q-Value for the Bellman update, SAC asks *both* Critics for their prediction, and takes the **Minimum** of the two.
$$ Q_{target} = \min(Q_1(s', a'), Q_2(s', a')) $$
By taking the minimum, SAC becomes pessimistic and avoids exploiting false optimistic spikes in the value predictions.

## 3. Entropy Maximization (To fix Exploration)
SAC totally changes the goal of Reinforcement Learning. Instead of just maximizing the expected return:
$$ J(\pi) = \sum r_t $$
SAC maximizes the expected return **PLUS its Entropy**:
$$ J(\pi) = \sum [r_t + \alpha \mathcal{H}(\pi(\cdot | s_t))] $$
*   **Entropy ($\mathcal{H}$)**: The measure of randomness/unpredictability.
*   **$\alpha$**: The temperature parameter that controls how much we care about exploration vs rewards.

This means if SAC sees two paths that both lead to the same reward, it will mathematically prefer the path that allows for the widest variety of actions, keeping its options open and safely exploring state spaces. 

## 4. The Squashed Gaussian Policy (The Reparameterization Trick)
Because we need entropy, the Actor can't be deterministic like DDPG. It must output a probability distribution.
1. The Actor outputs a `Mean` ($\mu$) and a `Log Standard Deviation` ($\log(\sigma)$).
2. We sample a random action from this Normal Distribution.
3. Because the action must be bounded between `[-1.0, 1.0]` for the physics engine, we pass the sampled action through a $\tanh$ function.
4. Finally, to let gradients flow through the random sampling operation, we use the **Reparameterization Trick**:
   $$ a = \tanh(\mu + \sigma \cdot \mathcal{N}(0, 1)) $$
