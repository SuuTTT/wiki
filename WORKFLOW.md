# Best Practice Workflow: Using LLMs to Write RL Research Code

Lessons learned from reimplementing DQN → PPO → SAC → Rainbow → DreamerV1 → Director in CleanRL style.

---

## The Core Problem

LLM-generated RL code has two failure modes:

1. **Large projects (Dreamer, Director):** 500-800 lines, ~5 independent bugs, doesn't run first time. Each bug takes 30min to diagnose because the code is unfamiliar.
2. **Small projects (DQN, PPO):** Runs fine but logs the wrong things, trains slowly, and you can't tell if it matches the paper.

Both stem from the same root cause: **no verification contract between you and the LLM.**

---

## The Layered Workflow

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 0: Specification (10 min)                        │
│  Write the metrics contract + acceptance criteria        │
├─────────────────────────────────────────────────────────┤
│  LAYER 1: Skeleton + Shape Tests (LLM generates)        │
│  Networks, buffers, env wrappers — no training loop      │
│  TEST: forward pass, backward pass, shapes correct       │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: Training Loop + Logging (LLM generates)       │
│  Main loop, loss computation, all metric logging         │
│  TEST: 1k-step smoke test — losses finite, logs exist    │
├─────────────────────────────────────────────────────────┤
│  LAYER 3: Full Run (you run)                             │
│  50k-200k steps on easy env                              │
│  TEST: learning curve matches paper reference numbers    │
├─────────────────────────────────────────────────────────┤
│  LAYER 4: Your Modification (LLM assists)                │
│  Research idea on top of verified baseline                │
│  TEST: A/B comparison against Layer 3                    │
└─────────────────────────────────────────────────────────┘
```

**Rule: Never skip a layer.** Each layer gates the next.

---

## Verification Stages

| Stage | Time | What You Catch |
|-------|------|----------------|
| **Shape test** | 2 sec | Wrong tensor dims, missing modules, import errors |
| **Smoke test** (1k steps) | 10 sec | NaN losses, dead gradients, reward off-by-one, action bugs |
| **Quick train** (50k steps) | 5 min | Wrong learning dynamics, missing log channels |
| **Full train** (200k-1M) | 20-60 min | Performance gap vs paper |

### The Smoke Test Template

Every generated file should include this or you add it yourself:

```python
if __name__ == "__main__" and args.smoke_test:
    # Override for fast verification
    args.total_timesteps = 1000
    args.learning_starts = 100
    args.batch_size = 32
    args.log_interval = 100
    # After training:
    assert not any(torch.isnan(p).any() for p in model.parameters()), "NaN in parameters"
    assert tracker.global_step == 1000, f"Step count wrong: {tracker.global_step}"
    print("SMOKE TEST PASSED")
```

---

## Acceptance Criteria: Know Your Target Numbers

Before writing ANY code, extract 3 checkpoints from the paper or known baselines.

### CleanRL Baselines (CartPole-v1)

| Algorithm | 50k steps | 200k steps | 500k steps |
|-----------|-----------|------------|------------|
| DQN | >300 | >450 | 500 |
| PPO | >400 | 500 | 500 |
| C51 | >350 | >450 | 500 |
| Rainbow | >400 | 500 | 500 |

### CleanRL Baselines (Pendulum-v1)

| Algorithm | 50k steps | 100k steps | 200k steps |
|-----------|-----------|------------|------------|
| DDPG | > -800 | > -400 | > -200 |
| SAC | > -600 | > -300 | > -200 |
| PPO-Continuous | > -800 | > -500 | > -300 |

### DreamerV1 Baselines (from paper)

| Env | 200k steps | 500k steps | 1M steps |
|-----|------------|------------|----------|
| Pendulum-v1 | > -400 | > -200 | > -150 |
| Walker-walk (DMC) | > 400 | > 800 | > 900 |
| Cheetah-run (DMC) | > 300 | > 500 | > 700 |

**Pin these in your prompt.** The LLM should log a comparison or at least you check against them.

---

## The Metrics Contract

Always specify exactly what to log. Here's the standard contract:

### For Any Algorithm

```
charts/episodic_return          ← rolling mean over last 10 episodes
charts/episodic_length          ← same
charts/SPS                      ← steps per second (speed check)
```

### For Value-Based (DQN, C51, Rainbow)

```
losses/td_loss                  ← temporal difference loss
losses/q_values                 ← mean Q-value (should grow, not explode)
charts/epsilon                  ← exploration rate decay
```

### For Policy Gradient (PPO, PPG)

```
losses/policy_loss              ← clipped surrogate loss
losses/value_loss               ← critic MSE
losses/entropy                  ← should decrease slowly, not collapse
charts/clipfrac                 ← fraction of clipped updates (~0.1-0.2)
charts/approx_kl                ← KL divergence between old and new policy
charts/explained_variance       ← how well critic explains returns (>0.5 is good)
```

### For Actor-Critic Continuous (DDPG, SAC)

```
losses/actor_loss               ← policy gradient loss
losses/critic_loss              ← Q-function TD loss
losses/alpha_loss               ← (SAC) entropy temperature loss
charts/alpha                    ← (SAC) current entropy coefficient
losses/q_values                 ← mean Q prediction
```

### For World Models (Dreamer)

```
losses/world_model              ← total WM loss
losses/kl                       ← KL divergence (raw, before clamp/free_nats)
losses/reconstruction           ← image/observation reconstruction loss
losses/reward_loss              ← reward prediction loss
losses/actor                    ← actor loss
losses/critic                   ← critic/value loss
debug/kl_raw                    ← KL before any clamping (catches free_nats masking)
debug/grad_norm                 ← gradient magnitude (catches exploding grads)
```

---

## Prompts

### Prompt 1: Small CleanRL Algorithm (DQN, PPO, SAC, etc.)

Use this for single-file algorithms on simple environments.

```
Implement {ALGORITHM} in CleanRL single-file style for {ENVIRONMENT}.

Requirements:
- Single .py file, all code self-contained
- Use tyro for CLI args, gymnasium for env, torch for networks
- Use RLTracker from cleanrl_utils.logger for all logging

Architecture:
- {describe networks, e.g. "QNetwork: 3-layer MLP (64,64), ReLU activations"}
- {describe buffer if needed, e.g. "Replay buffer: 10k capacity, uniform sampling"}

Training:
- total_timesteps: {N}
- learning_rate: {lr}
- batch_size: {bs}
- {other key hyperparams from paper}

Logging contract — log ALL of these via tracker.log_metrics():
  {paste the relevant section from the metrics contract above}

Log episodic_return and episodic_length via tracker.log_episode() on every episode end.
Log SPS via tracker.log_sps() every 100 global steps.

Include a smoke test mode: when --smoke-test is passed, override total_timesteps=1000,
learning_starts=100, and assert no NaN in parameters at the end.

Acceptance criteria (do NOT need to assert, but keep in mind):
- {ENV} should reach return > {X} by {N} steps
- SPS should be > {Y} on this hardware

Reference: {paper name and year}
```

**Example filled in:**

```
Implement DQN in CleanRL single-file style for CartPole-v1.

Requirements:
- Single .py file, all code self-contained
- Use tyro for CLI args, gymnasium for env, torch for networks
- Use RLTracker from cleanrl_utils.logger for all logging

Architecture:
- QNetwork: 3-layer MLP (120, 84), ReLU activations
- Replay buffer: 10k capacity, uniform sampling

Training:
- total_timesteps: 500000
- learning_rate: 2.5e-4
- batch_size: 128
- target_network_frequency: 500
- epsilon: linear decay from 1.0 to 0.05 over first 50% of training
- gamma: 0.99, tau: 1.0

Logging contract — log ALL of these via tracker.log_metrics():
  losses/td_loss, losses/q_values, charts/epsilon

Log episodic_return and episodic_length via tracker.log_episode() on every episode end.
Log SPS via tracker.log_sps() every 100 global steps.

Include a smoke test mode: when --smoke-test is passed, override total_timesteps=1000,
learning_starts=100, and assert no NaN in parameters at the end.

Acceptance criteria:
- CartPole-v1 should reach return > 450 by 200k steps
- SPS should be > 500

Reference: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
```

---

### Prompt 2: Large World Model Project (Dreamer, Director)

Use this for multi-component systems. **Generate in layers, not all at once.**

#### Layer 1 — World Model Only

```
Implement the world model component of {ALGORITHM} in CleanRL single-file style.

This is LAYER 1 of a multi-layer build. Only implement:
- RSSM (Recurrent State Space Model)
  - Deterministic state: {deter_size}-dim GRU
  - Stochastic state: {stoch description, e.g. "30-dim diagonal Gaussian" or "32x32 categorical"}
  - observe() method: takes real observations, returns posterior + prior
  - imagine() method: given start states + policy, rolls forward without observations
- Encoder: {description, e.g. "4-layer CNN for 64x64 images" or "MLP for vector obs"}
- Decoder: {description}
- Reward head: MLP predicting reward from latent state
- Continue head: MLP predicting episode continuation (Bernoulli)

Do NOT implement actor, critic, or training loop yet.

Include a shape test at the bottom:
```python
if __name__ == "__main__":
    # Test all shapes with dummy data
    batch, seq_len = 4, 50
    obs = torch.randn(batch, seq_len, *obs_shape)
    actions = torch.randn(batch, seq_len, act_dim)
    # ... test observe, imagine, decode, reward, continue
    print("ALL SHAPE TESTS PASSED")
```

RSSM KL loss: {describe, e.g. "single KL with free_nats=3.0" or "balanced KL 0.8/0.2 with AutoAdapt target=3.5"}

Reference: {paper}, Section {N}
```

#### Layer 2 — Actor-Critic in Imagination

```
Continuing from the world model in {FILE}, now add:

- Actor: MLP that maps latent state → action distribution
  - For continuous: output mean + std of a TanhNormal distribution
  - Gradient: {backprop through world model / REINFORCE}
- Critic: MLP that maps latent state → scalar value
  - Target: lambda-returns with lambda={0.95}, horizon={15}
  - Slow target network updated every {100} steps
- Imagination training:
  1. Sample batch from replay buffer
  2. Run RSSM.observe() to get posterior states
  3. From random timesteps, run RSSM.imagine() for {H} steps using actor
  4. Compute lambda-returns from imagined rewards and values
  5. Update actor to maximize returns, critic to predict returns

Logging contract for this layer:
  losses/actor, losses/critic, losses/world_model, losses/kl, losses/reconstruction,
  losses/reward_loss, debug/kl_raw, debug/grad_norm

Include smoke test: 1k steps total, assert losses are finite and decreasing.

Reference: {paper}, Section {N}
```

#### Layer 3 — Full Training Loop

```
Continuing from {FILE}, now add the main training loop:

- Environment interaction:
  - Use the RSSM + encoder + actor for action selection
  - Store transitions in replay buffer (capacity: {1M}, chunk_length: {50})
  - Collect {seed_episodes} random episodes before training
- Training schedule:
  - Every {train_every} env steps, do {train_steps} gradient updates
  - Each update: sample batch, train world model, then train actor-critic
- Episode tracking:
  - On each episode end, log return and length via tracker.log_episode()
  - Log SPS every {1000} steps
- Video recording:
  - Every {log_every_n_episodes} episodes, record a gym video

Hyperparameters:
  total_timesteps: {N}, learning_rate: {lr}, batch_size: {bs}
  replay_capacity: {cap}, seed_episodes: {seed}, train_every: {N}, train_steps: {N}

The file should be fully runnable: `python {filename} --env-id Pendulum-v1`

Acceptance criteria:
- Pendulum-v1: return > -400 by 100k steps, > -200 by 200k steps
- SPS > {expected} on this hardware
```

---

### Prompt 3: Adding Hierarchy (Director, Options, HIRO)

Only use AFTER the flat version (Dreamer) is verified and working.

```
Starting from the verified {BASE_ALGORITHM} in {FILE}, add hierarchical control:

Manager:
- Operates every K={8} steps
- Action space: {describe, e.g. "8x8 one-hot categorical skill codes"}
- Reward: {describe, e.g. "1.0 * extrinsic + 0.1 * exploration"}
- Gradient: {REINFORCE / backprop}

Worker:
- Operates every step
- Additional inputs: {describe, e.g. "goal (1024-dim) + delta (goal - current_deter)"}
- Reward: {describe, e.g. "cosine similarity between state and goal"}
- Gradient: {backprop / REINFORCE}

Goal Autoencoder (if applicable):
- Encoder: state → skill code distribution
- Decoder: skill code → goal vector
- Loss: reconstruction + KL to uniform prior
- Train on: replay data

Key changes to imagination:
- imagine_carry(): carry Manager state (step counter, skill, goal) across steps
- split_traj(): chop trajectory into K-step chunks for Worker
- abstract_traj(): subsample every K-th state for Manager

Logging additions:
  losses/manager_actor, losses/worker_actor, losses/manager_critic, losses/worker_critic,
  losses/goal_vae, metrics/goal_reward_mean, metrics/skill_entropy

Smoke test: 1k steps, assert Manager and Worker losses are both finite.

Reference: {paper}
```

---

### Prompt 4: Research Modification on Verified Baseline

```
I have a verified working implementation of {ALGORITHM} in {FILE}.
It achieves return > {X} on {ENV} at {N} steps (confirmed).

I want to test this modification:
  {describe your idea in 2-3 sentences}

Implementation plan:
  {describe what to change — which function, what the new computation is}

Constraints:
- Change ONLY what's needed for the modification. Do not refactor, rename, or "improve" anything else.
- Add a CLI flag --{flag-name} (default: False) to toggle the modification on/off
  so I can A/B test against the baseline with the same file.
- Log any new metrics under "research/{metric_name}" namespace.

I will test by running:
  python {file} --env-id {ENV} --total-timesteps {N}           # baseline
  python {file} --env-id {ENV} --total-timesteps {N} --{flag}  # modification

Expected comparison: {what you expect to see — faster convergence, higher return, etc.}
```

---

### Prompt 5: Debug a Specific Bug

When training looks wrong, use this targeted prompt:

```
I'm training {ALGORITHM} on {ENV}. Here's what I see after {N} steps:

Symptom:
  {describe exactly what's wrong, e.g. "episodic_return stuck at -1200, never improving"}

Relevant metrics:
  losses/kl: {value, e.g. "stuck at 3.0 (exactly free_nats)"}
  losses/actor: {value}
  losses/critic: {value}
  charts/SPS: {value}

Here's the relevant code section:
```python
{paste the 20-50 lines most likely to contain the bug}
```

I suspect the issue might be in: {your hypothesis}

Please:
1. Diagnose the most likely root cause
2. Show the exact fix (minimal diff)
3. Explain why this bug produces the observed symptom
```

---

### Prompt 6: Explain an Algorithm Before Implementing

Use this BEFORE writing code to build understanding.

```
I'm about to implement {ALGORITHM} ({paper title, year}).

Before writing code, explain every design choice and trick in the paper
using the Feynman method — as if teaching someone who understands basic
neural networks and RL (policy gradient, Q-learning, actor-critic) but
has never seen this specific algorithm.

For each component, cover:
1. What problem does this solve? (What goes wrong without it?)
2. How does it work? (Intuitive explanation + exact equation)
3. What are the hyperparameters and their typical values?
4. Common implementation pitfalls

Structure as numbered sections. Include a summary table at the end mapping
every design choice → its value → why.

Reference code to ground the explanations: {path or repo}
```

---

## Common Pitfalls Checklist

Check these before starting a full training run:

### Off-by-one Errors (Dreamer-style)
- [ ] Reward at time $t$ corresponds to the transition $s_t \to s_{t+1}$, not $s_{t-1} \to s_t$
- [ ] Lambda returns: the last value in the horizon uses the critic, not reward
- [ ] `obs[0]` is the initial observation BEFORE any action is taken

### Action Bugs
- [ ] `prev_action` is actually the previous action, not always zeros
- [ ] Action is stored in replay BEFORE being passed to env.step()
- [ ] Continuous actions are properly squashed (tanh) and un-squashed (atanh) for log-prob

### Logging Bugs
- [ ] `global_step` increments by `num_envs` per vectorized step, not by 1
- [ ] Episode return is the REAL cumulative reward, not a partial window
- [ ] KL is logged RAW (before free_nats clamp) — otherwise you can't tell if the model learned

### Speed Bugs
- [ ] `torch.no_grad()` around all inference (env interaction, target network)
- [ ] Imagination horizon isn't unnecessarily long
- [ ] No Python loops where vectorized ops work (e.g., lambda returns as a scan, not a for-loop)

### World Model Bugs
- [ ] RSSM observe() uses teacher forcing (real observations), imagine() does not
- [ ] KL is computed between posterior and prior, not prior and prior
- [ ] Reconstruction loss targets are stop-gradiented observations, not latent states
- [ ] Continue model predicts $\gamma$ from the STATE, reward model predicts from the STATE too

---

## Project Lifecycle at a Glance

```
Week 1: Read paper → Prompt 6 (explain) → understand every component
         ↓
Week 1: Prompt 1 or 2-Layer1 (skeleton) → shape tests pass
         ↓
Day 2:  Layer 2 (training) → smoke test passes
         ↓
Day 2:  Layer 3 (full loop) → quick train matches paper on easy env
         ↓
Day 3:  Full training run → acceptance criteria met
         ↓
Day 4+: Prompt 4 (your modification) → A/B test
         ↓
         Research result
```

For simple algorithms (DQN, PPO): Layers 1-3 collapse into a single Prompt 1 call.
For complex algorithms (Dreamer, Director): Always do layers separately.

---

## File Organisation

```
wiki/
├── WORKFLOW.md                    ← this file
├── learn-cleanrl/
│   ├── dqn_tutorial.py            ← simple, single prompt
│   ├── ppo_tutorial.py
│   ├── sac_tutorial.py
│   ├── ...
│   └── docs/
│       ├── DQN.md                 ← theory wiki per algorithm
│       └── PPO.md
└── research-world-models/
    ├── phase4_dreamer/
    │   ├── 04_dreamer_claude.py   ← layered build: WM → AC → loop
    │   └── DREAMER_EXPLAINED.md   ← Prompt 6 output
    └── phase5_director/
        ├── 05_director_claude.py  ← layer on top of verified Dreamer
        └── DIRECTOR_EXPLAINED.md
```

---

## Quick Reference: Which Prompt to Use

| Situation | Prompt |
|-----------|--------|
| New simple algorithm (DQN, PPO, C51, SAC) | **Prompt 1** |
| New complex algorithm — world model piece | **Prompt 2, Layer 1** |
| New complex algorithm — actor-critic piece | **Prompt 2, Layer 2** |
| New complex algorithm — full training loop | **Prompt 2, Layer 3** |
| Adding hierarchy to existing algorithm | **Prompt 3** |
| Testing my own research idea | **Prompt 4** |
| Something is broken during training | **Prompt 5** |
| Understanding a paper before coding | **Prompt 6** |
