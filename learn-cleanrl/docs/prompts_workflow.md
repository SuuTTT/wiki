# CleanRL MuJoCo Implementation Prompts

Following the [Best Practice Workflow](../WORKFLOW.md), use these prompts sequentially to implement and verify the MuJoCo learning path.

---

## LAYER 0: Theoretical Specification
**Use this to build intuition before writing code.**

```markdown
I'm about to implement PPO for continuous control (MuJoCo) based on the CleanRL style.

Before writing code, explain the "Continuous Logic" using the Feynman method:
1. What math (probability distributions) is used to parameterize actions vs. discrete actions?
2. How does the Clipped Surrogate Objective change when dealing with log-probabilities of Gaussian distributions?
3. What is the role of the "Entropy Bonus" in a continuous space, and how is it calculated?
4. What are the critical hyperparameters for MuJoCo (e.g., target_kl, norm_obs, norm_ret)?

Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
```

---

## LAYER 1: Skeleton & Shape Tests (Architecture)
**Use this to generate the network and buffer structure.**

```markdown
Implement the architecture for PPO Continuous (MuJoCo) in a CleanRL-style script.

Requirement: Only implement the Class structures and Shape tests. No training loop yet.
- Agent Class: 
    - Critic: 3-layer MLP (64, 64), Tanh activations, scalar output.
    - Actor: 3-layer MLP (64, 64), Tanh activations. 
    - Output: mean (mu) and log_std for action distribution.
- Environment: Use gymnasium with 'HalfCheetah-v4'.
- Wrappers: Include `ClipAction`, `NormalizeObservation`, `TransformObservation`, `NormalizeReward`, and `TransformReward`.

Include a shape test block:
if __name__ == "__main__":
    # Test: pass a dummy observation through actor and critic
    # Assert: actor output shape matches action space, critic is (1,)
    print("LAYER 1: SHAPE TESTS PASSED")
```

---

## LAYER 2: Training Loop & Logging (The "Meat")
**Use this to build the functional training script.**

```markdown
Complete the PPO Continuous implementation in {FILE_FROM_LAYER_1}.

Add the Training Loop:
- total_timesteps: 1,000,000
- update_epochs: 10, minibatch_size: 64
- gamma: 0.99, gae_lambda: 0.95
- target_kl: 0.015

Logging contract (use RLTracker):
- charts/episodic_return, charts/episodic_length, charts/SPS
- losses/value_loss, losses/policy_loss, losses/entropy
- charts/approx_kl, charts/clipfrac, charts/explained_variance

Include SMOKE TEST mode:
When --smoke-test is passed:
1. total_timesteps = 2000
2. rollout_steps = 128
3. Assert no NaN in policy gradients.
```

---

## LAYER 3: Performance Validation (The Audit)
**Use this to verify against "True Performance" extracted from the paper.**

```markdown
I have finished training PPO on Pendulum-v1 and HalfCheetah-v4.
Local Results:
- Pendulum-v1: {LOCAL_RETURN}
- HalfCheetah-v4: {LOCAL_RETURN}

Paper Baselines (arXiv:2205.12740):
- Pendulum-v1: -150 to -200
- HalfCheetah-v4: 3000+

Task:
1. Compare my local episodic_return curve against these baselines.
2. If I am underperforming, check the "Observed Issues" in /workspace/wiki/learn-cleanrl/VALIDATION.md.
3. Verify if `explained_variance` is > 0.5; if not, suggest improvements to the Critic network.
```

---

## LAYER 4: Research Modification (World Model/HRL/GNN)
**Use this for your specific research idea.**

```markdown
I have a verified PPO baseline for MuJoCo. I want to modify it for {RESEARCH_TOPIC}.

Proposed Idea: {YOUR_IDEA, e.g., "Add a World Model to predict next states and use 'imagination' for extra updates."}

Requirement:
1. Implement the {COMPONENT} (e.g., RSSM or GNN layer).
2. Create an A/B test script that runs both the baseline and the modification.
3. Log `debug/latent_loss` and `debug/reconstruction_error` to verify the modification is learning.
```
