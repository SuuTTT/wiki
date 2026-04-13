# Proximal Policy Optimization (PPO)

Welcome to PPO. While DQN learns the *Value* of an action and taking the max (Value-Based), PPO directly learns the *Policy* (the probability distribution of actions) to maximize the expected return.

## 1. Actor-Critic Architecture
PPO uses two separate networks (or a shared network with two heads):
1.  **The Actor**: Outputs a probability distribution over actions (e.g., logits for a `Categorical` distribution in discrete spaces, or mean/std for a `Normal` distribution in continuous spaces).
2.  **The Critic**: Outputs a single scalar: the expected return (Value) of the current state, exactly like a DQN but without specifying the action.

## 2. Generalized Advantage Estimation (GAE)
We need to know if an action was *better than expected*.
*   **Advantage ($A_t$)**: The difference between the actual return and the Critic's predicted Value ($V(s)$).
    *   $A_t > 0$: Action was unexpectedly good. Increase its probability.
    *   $A_t < 0$: Action was unexpectedly bad. Decrease its probability.

GAE smooths out the variance by blending 1-step, 2-step, and N-step advantages using a decay parameter $\lambda$ (`gae_lambda`).
$$ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $$
$$ A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + ... = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $$

## 3. The PPO Clipped Surrogate Objective
In classic Policy Gradient, a good action might trigger a massive gradient update that destroys the policy, causing learning to collapse.
PPO fixes this by taking the ratio between the *new* policy and the *old* policy:
$$ r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} $$

If $r_t(\theta) > 1$, the action is more likely now than before.
Instead of updating the policy infinitely, PPO clips the ratio to a small range (usually $[1 - \epsilon, 1 + \epsilon]$, where $\epsilon=0.2$).

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right] $$

This ensures that the policy doesn't change too much in a single update, leading to highly stable, monotonic improvement.

## 4. Why On-Policy?
Unlike DQN, which saves experiences to a Replay Buffer and trains on old data (Off-Policy), PPO requires data generated *by the current policy* (On-Policy). PPO rolls out the agent for $N$ steps, calculates advantages, updates the network, and then throws the data away.