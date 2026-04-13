# Phasic Policy Gradient (PPG)

In standard Proximal Policy Optimization (PPO), there is a classic conflict known as **Feature Interference**. 

Standard PPO uses a single shared neural network body (Feature Extractor) that branches out into two heads:
1. **The Actor Head**: Outputs actions (What should I do?)
2. **The Critic Head**: Outputs state values (How good is this situation?)

The loss function combines both: `total_loss = policy_loss + 0.5 * value_loss`.
**The Problem**: The Value Loss is often mathematically massive and noisy compared to the Policy Loss. When they backpropagate down into the shared feature extractor, the Value gradients overpower the Policy gradients, destroying the fragile features the Actor needed to make precise, clever decisions.

## 1. Disjoint Networks
The simplest solution is to just use two completely separate Neural Networks. One for the Actor, one for the Critic.
**The New Problem:** In purely visual environments (like complex images), the Actor and Critic don't get to share the visual filters they've learned. The Actor has to learn what a wall is from scratch, and the Critic has to learn what a wall is from scratch. This makes the algorithm incredibly sample-inefficient.

## 2. Phasic Policy Gradient
PPG solves this by separating the **Training Phase**, rather than keeping the networks permanently disjoint.
It introduces two distinct phases of training:

### Phase 1: The Policy Phase
* The Actor is trained specifically to maximize Advantages (PPO clip).
* The Critic is trained specifically to predict Returns (MSE loss).
* They do not share gradients here! The Actor focuses *only* on acting.

### Phase 2: The Auxiliary Phase (The Feature Sharing Phase)
After running Phase 1 a few times, PPG forces the networks to share information.
* We freeze the Critic's accurate value predictions as "True Targets".
* We add an **Auxiliary Value Head** to the *Actor's network*.
* In Phase 2, the Actor is trained to predict the same values the Critic just predicted, using MSE loss.
* To prevent the Actor from messing up its own policy while doing this, we add a **KL-Divergence penalty**, forcing its action probabilities to stay exactly the same as they were before the Auxiliary Phase began.

This gives you the best of both worlds: Perfect gradient separation during acting, and perfect feature sharing during resting.