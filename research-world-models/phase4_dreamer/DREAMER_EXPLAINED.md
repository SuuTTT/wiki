# DreamerV1: Every Design Choice Explained from Scratch

*A Feynman-style explanation — if you can't explain it simply, you don't understand it well enough.*

---

## Table of Contents

1. [The Big Picture: What Problem Are We Solving?](#1-the-big-picture)
2. [Why "Dream"? The Core Insight](#2-why-dream)
3. [The World Model — Teaching a Neural Net to Simulate Reality](#3-the-world-model)
4. [Prior and Posterior — Demystified with a Weather Analogy](#4-prior-and-posterior)
5. [The RSSM — Why Two Kinds of Memory?](#5-the-rssm)
6. [The KL Divergence Loss — Why Compare Prior and Posterior?](#6-the-kl-divergence-loss)
7. [Free Nats — A Deliberate "Slack" Knob](#7-free-nats)
8. [The Encoder and Decoder — Compressing Pixels](#8-the-encoder-and-decoder)
9. [The Actor — Learning to Act in a Dream](#9-the-actor)
10. [The Critic — Estimating "How Good Is This?"](#10-the-critic)
11. [Lambda Returns — Balancing Bias and Variance](#11-lambda-returns)
12. [The Three Training Phases](#12-the-three-training-phases)
13. [Exploration Noise](#13-exploration-noise)
14. [Action Repeat](#14-action-repeat)
15. [Tanh-Squashed Normal — Why Not Just a Normal Distribution?](#15-tanh-squashed-normal)
16. [The Reparameterisation Trick — Backprop Through Randomness](#16-the-reparameterisation-trick)
17. [Episode-Based Replay Buffer](#17-episode-based-replay-buffer)
18. [Putting It All Together — The Full Algorithm](#18-putting-it-all-together)
19. [Summary Table of Every Design Choice](#19-summary-table)

---

## 1. The Big Picture

Imagine you want to teach a robot to drive a car by looking at camera images. The robot sees pixels (a 64×64 RGB image), takes an action (steer, gas, brake), gets a reward (did it stay on the road?), and sees a new image.

**The naive approach:** Try random actions, see what happens, update a policy. This is model-free RL (like PPO or SAC). It works, but it's *extremely* wasteful — you need millions of real interactions because you learn nothing about *how the world works*. Every piece of knowledge is locked inside neural network weights with no explicit structure.

**The Dreamer approach:** First, *learn how the world works* (a "world model"). Then, *practice inside your own imagination* instead of the real world. This is called **model-based RL**.

The analogy: A chess grandmaster doesn't need to play millions of games. They build a mental model of the board, then *think ahead* ("if I move here, they'll probably move there, then I can..."). Dreamer does exactly this — but for continuous control from pixels.

---

## 2. Why "Dream"?

The name comes from the key insight:

> **You can train a policy entirely inside the imagined futures of your world model, without touching the real environment.**

Here's the training loop at the highest level:

```
Repeat forever:
  1. Interact with the real environment, store experience
  2. Use stored experience to improve the world model
  3. "Dream" — imagine thousands of future trajectories using the world model
  4. Train the actor (policy) to maximise reward in those dreams
  5. Train the critic to estimate expected reward in those dreams
```

Why is this powerful?

- **Data efficiency:** One real experience can be replayed and "dreamed about" many times
- **Planning:** The actor can explore consequences of actions before committing
- **Speed:** Imagination happens on GPU — no physics simulator bottleneck

---

## 3. The World Model — Teaching a Neural Net to Simulate Reality

The world model must answer one question:

> **Given the current state and an action, what happens next?**

But there's a catch: the "state" of the world isn't directly observable. You see *pixels*, not the true underlying state (car velocity, road curvature, tire grip, etc.). So the world model has two jobs:

1. **Infer** a compact latent state from pixel observations (perception)
2. **Predict** how that latent state evolves over time (dynamics)

This is where the RSSM comes in.

---

## 4. Prior and Posterior — Demystified with a Weather Analogy

These terms come from probability theory, but the intuition is dead simple.

### The Weather Analogy

Imagine you want to predict tomorrow's weather:

**Prior** = Your prediction *before* looking out the window.
You know it's April in London, so you might guess: "60% chance of rain, 40% chance of sun." This is based purely on your *model of how weather works* — patterns, seasons, yesterday's weather. You haven't looked at any new evidence yet.

**Posterior** = Your prediction *after* looking out the window.
You glance outside and see dark clouds gathering. Now you update: "90% chance of rain, 10% chance of sun." The observation (clouds) gave you extra information that refined your belief.

**Key insight: The posterior is always more accurate than the prior, because it has access to more information.**

### In Dreamer

The world has a hidden state $s_t$ at each time step. We never see it directly — we only see pixels.

| | What it represents | What information it uses | When it's used |
|---|---|---|---|
| **Prior** $p(s_t \mid h_t)$ | "What I *predict* the state is, based only on my dynamics model" | Previous state + previous action → GRU → predicted state. **No observation.** | During imagination (dreaming). We don't have real observations in a dream. |
| **Posterior** $q(s_t \mid h_t, o_t)$ | "What the state *actually is*, after I also look at the observation" | Prior's dynamics + the actual observation at time $t$. | During training on real data. We have the real observation, so we should use it. |

$h_t$ is the deterministic memory of the GRU (more on this in §5).

### Why Do We Need Both?

- **During training:** We have real observations, so we use the posterior (more accurate) to build the latent states. This gives us better training signal for the decoder, reward model, and critic.
  
- **During imagination:** We don't have real observations (we're dreaming!), so we can only use the prior. The prior says "based on my understanding of physics, the next state should be around here."

- **The training objective:** We want the prior to become as good as the posterior. If the prior matches the posterior perfectly, our dynamics model has fully captured how the world works — it no longer *needs* the observation to know what's going on.

This leads directly to the KL divergence loss (§6).

### A Concrete Example

The car is driving. At time $t$:

1. **GRU processes** the previous state and action → deterministic state $h_t$
2. **Prior says:** "Based on dynamics, the stochastic state $s_t$ should be around $\mathcal{N}(\mu_{\text{prior}}, \sigma_{\text{prior}})$"
   - *"I think the car is probably turning left at medium speed"*
3. **We look at the actual frame** $o_t$ → encoder produces embedding $e_t$
4. **Posterior says:** "After seeing the frame, $s_t$ should be around $\mathcal{N}(\mu_{\text{post}}, \sigma_{\text{post}})$"
   - *"Oh, the car is actually turning left but going faster than I thought"*
5. **KL loss** pushes the prior's $(\mu, \sigma)$ to match the posterior's $(\mu, \sigma)$
   - *"Next time, predict higher speed when turning left on this kind of road"*

---

## 5. The RSSM — Why Two Kinds of Memory?

RSSM = **Recurrent State-Space Model**. It's the heart of Dreamer.

### The Problem with Pure RNNs

A vanilla RNN/GRU maintains a hidden state $h_t$ that encodes everything — it's entirely deterministic. Given the same inputs, you always get the same $h_t$.

But the real world is **stochastic** (random). If you drop a ball, it might bounce slightly left or slightly right depending on micro-variations in the surface. A purely deterministic model can't represent this uncertainty — it would predict the *average* of all possible outcomes, which might be something that *never actually happens* (the ball splits into two phantom trajectories).

### The Solution: Split the State in Two

The RSSM maintains two kinds of state:

#### 1. Deterministic state $h_t$ (the GRU hidden state)
- Size: 200 dimensions
- Updated by: $h_t = \text{GRU}(f(s_{t-1}, a_{t-1}), h_{t-1})$
- Purpose: **Long-range memory**. Like the RNN backbone — remembers things over many steps. "The car has been on a straight road for the last 20 frames."

#### 2. Stochastic state $s_t$ (sampled from a Gaussian)
- Size: 30 dimensions  
- Sampled from: $s_t \sim \mathcal{N}(\mu(h_t), \sigma(h_t))$
- Purpose: **Encoding uncertainty**. Represents the random aspects of the current moment. "The exact road texture and micro-position right now."

#### The full state (called "feature")
$$z_t = [h_t; s_t] \in \mathbb{R}^{230}$$

Everything downstream (decoder, reward predictor, actor, critic) sees this 230-dimensional feature.

### Why Not Just Stochastic? Why Not Just Deterministic?

| Model type | Problem |
|---|---|
| Pure deterministic (GRU only) | Can't represent uncertainty. Collapses multi-modal futures into blurry averages. |
| Pure stochastic (VAE per step) | Information must pass through a sampling bottleneck at every step. Long-range memory is destroyed — each step "forgets" through the noise. |
| **RSSM (both)** | The deterministic path carries reliable long-range memory. The stochastic part captures per-step uncertainty. Best of both worlds. |

### The Information Flow

```
Time t-1                          Time t
────────                          ──────

 s_{t-1}  ─┐
            ├─ concat ─ Dense ─ ELU ─┐
 a_{t-1}  ─┘                        │
                                     ▼
                           GRU(input, h_{t-1}) ──────► h_t  (deterministic)
                                                        │
                                                        ▼
                                              ┌─── Dense ─ ELU ─ Dense
                                              │
                                          μ_prior, σ_prior         ← PRIOR
                                              │
                                    s_t ~ N(μ_prior, σ_prior)      (if imagining)
                                              
                                              OR

                              h_t + e_t (encoder embedding)
                                 ├─ Dense ─ ELU ─ Dense
                                 │
                            μ_post, σ_post                         ← POSTERIOR  
                                 │
                       s_t ~ N(μ_post, σ_post)                     (if training on real data)
```

---

## 6. The KL Divergence Loss — Why Compare Prior and Posterior?

### What Is KL Divergence?

KL divergence measures **how different two probability distributions are**. If $P$ and $Q$ are two distributions:

$$D_{\text{KL}}(Q \| P) = \mathbb{E}_{x \sim Q}\left[\log \frac{Q(x)}{P(x)}\right]$$

- If $Q = P$: $D_{\text{KL}} = 0$ (identical distributions)
- If $Q \neq P$: $D_{\text{KL}} > 0$ (the more different, the larger)

For two Gaussians (which is what we have), there's a clean closed-form formula — no sampling needed.

### Why Use KL in Dreamer?

Remember:
- **Posterior** $q(s_t | h_t, o_t)$ = "what the state is after seeing the observation" (accurate)
- **Prior** $p(s_t | h_t)$ = "what the state is predicted to be, without the observation" (our dynamics model)

We want the dynamics model (prior) to become so good that it doesn't *need* the observation. So we minimise:

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(\text{posterior} \| \text{prior})$$

This says: **"Make the prior's prediction match the posterior's informed belief."**

In code:
```python
post_dist  = Normal(posts["mean"],  posts["std"])
prior_dist = Normal(priors["mean"], priors["std"])
kl = kl_divergence(post_dist, prior_dist).sum(dim=-1)  # sum over 30 stoch dims
```

### Intuition: The KL Loss Trains the Dynamics Model

When KL is high → the prior is far from the posterior → the dynamics model made a bad prediction about what would happen → big loss → gradient update makes the prediction better.

When KL is low → the dynamics model predicted almost exactly what the observation confirmed → the world model is accurate.

---

## 7. Free Nats — A Deliberate "Slack" Knob

### The Problem

If we push KL all the way to zero, the posterior *collapses* to the prior. This means the stochastic state $s_t$ ignores the observation and just copies the dynamics prediction. The model becomes deterministic in disguise, and the decoder/reward model lose useful information.

Think of it this way: the KL loss says "don't encode too much in $s_t$." The reconstruction loss says "encode enough in $s_t$ to reconstruct the image." These goals fight each other. If KL wins too hard, $s_t$ becomes useless.

### The Solution: free_nats = 3.0

```python
kl_loss = torch.clamp(kl, min=free_nats).mean()
```

This means: **"Don't penalise the KL if it's already below 3 nats."**

- If raw KL = 1.5 → clamped to 3.0 → gradient is zero → the model is free to use the stochastic state
- If raw KL = 8.0 → loss is 8.0 → gradient pushes the prior toward the posterior

The name "free nats" literally means "these nats of information are free — no penalty." It gives the posterior room to encode useful details that the prior can't easily predict (random texture variations, exact pixel noise, etc.).

### The "Stuck at 3.0" Symptom

If you see `kl_loss = 3.0000` in your logs, it means the actual KL is *below* 3 nats everywhere — the clamp is active. This is normal and healthy early in training. It means the dynamics model is already very close to the posterior, and the remaining information difference is allowed to exist penalty-free.

---

## 8. The Encoder and Decoder — Compressing Pixels

### Encoder: Pixels → Compact Embedding

```
Input:   (B, 3, 64, 64)  — RGB image
Conv1:   (B, 32, 31, 31)   kernel=4, stride=2
Conv2:   (B, 64, 14, 14)
Conv3:   (B, 128, 6, 6)
Conv4:   (B, 256, 2, 2)
Flatten: (B, 1024)
```

Each convolution halves the spatial size while doubling the channels. This is a classic pattern (depth ×1, ×2, ×4, ×8). The output is a 1024-dimensional vector that captures the essence of the 64×64×3 = 12,288 pixel image — a **87× compression**.

The encoder output feeds into the posterior (it's the observation signal $e_t$).

### Decoder: Latent State → Reconstructed Pixels

Does the inverse: takes the 230-dim feature $z_t$, projects it to 1024, reshapes to (1024, 1, 1), then runs 4 transposed convolutions back up to (3, 64, 64).

The reconstruction loss (MSE between predicted and actual pixels) is what forces the latent state to contain *enough information* to recreate the observation. Without it, the latent state could ignore the pixels entirely.

### Why MSE and not something fancier?

MSE loss on pixel values is equivalent to maximising the log-likelihood under a Gaussian observation model with fixed variance σ=1:

$$-\log p(o_t | z_t) = \frac{1}{2\sigma^2}\|o_t - \hat{o}_t\|^2 + \text{const}$$

With σ=1, this is just MSE (up to a constant). It's simple and works well enough — the decoder doesn't need to produce perfect images, just "good enough" that the latent state is forced to encode useful information.

---

## 9. The Actor — Learning to Act in a Dream

This is where Dreamer gets its power. The actor is trained **entirely in imagination**.

### The Process

1. Take all the posterior states from the real training batch (B×T = 2500 states)
2. From each state, imagine H=15 steps into the future using the world model
3. At each imagined step, the actor chooses an action
4. The reward model predicts the reward at each imagined step
5. Compute λ-returns (§11) — "how much total reward do I expect from here?"
6. **Actor loss = negative of discounted λ-returns** (because we want to maximise reward)

### Why This Works

The actor's actions flow through the world model, which produces imagined states, which produce imagined rewards. The entire chain is differentiable:

```
actor(z_t) → action_t → RSSM.img_step(state_t, action_t) → state_{t+1} → reward_model(z_{t+1}) → reward_{t+1}
```

We can compute $\frac{\partial \text{total\_reward}}{\partial \theta_{\text{actor}}}$ directly via backpropagation! This is much more efficient than policy gradient methods (REINFORCE, PPO) which need thousands of real trajectories to estimate this gradient.

### The Discount Weighting

Imagined steps further in the future are less reliable (errors compound). So we apply exponential discounting:

$$\mathcal{L}_{\text{actor}} = -\frac{1}{N}\sum_{t=0}^{H-2} \gamma^t \cdot V_\lambda(t)$$

Step 0 (right after the real data) gets weight $\gamma^0 = 1$. Step 14 gets weight $\gamma^{14} = 0.99^{14} \approx 0.87$.

### Why Actor Loss Is a Large Negative Number

If the mean λ-return across imagination is ~100, then `actor_loss ≈ -100`. This is not a bug — it's a maximisation objective expressed as minimisation (we minimise the negative). The actor is "happy" when the loss is very negative (high expected reward). It would only be zero if the expected reward was zero.

---

## 10. The Critic — Estimating "How Good Is This?"

The value model (critic) predicts:

$$v(z_t) \approx \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k}\right]$$

"Starting from state $z_t$, how much total discounted reward do I expect?"

### Why Do We Need a Critic?

We can only imagine H=15 steps ahead (not infinity). At step 15, we need to estimate "how much reward is there beyond this horizon?" The value function provides this **bootstrap**:

$$V_\lambda(H-1) = r_{H-1} + \gamma \cdot v(z_H)$$

Without the critic, we'd either need to imagine infinitely far (impossible) or cut off and lose all future reward signal (bad).

### How It's Trained

The critic is trained to predict the λ-returns (which are better estimates than raw value predictions because they use actual imagined rewards for the first H steps):

$$\mathcal{L}_{\text{value}} = \mathbb{E}\left[\|v(z_t) - V_\lambda(t)\|^2\right]$$

Important: the λ-return targets are **detached** (no gradient flows through them into the actor or world model). The value model trains independently to be an accurate predictor.

### Architecture

A 3-layer MLP with ELU activations:
```
z_t (230 dim) → Dense(400) → ELU → Dense(400) → ELU → Dense(400) → ELU → Dense(1)
```

3 layers for the value model (vs 2 for the reward model) because value prediction is harder — it must estimate long-horizon cumulative reward, not just immediate reward.

---

## 11. Lambda Returns — Balancing Bias and Variance

### The Problem

When training the actor in imagination, we need a "target" — how good is each imagined state? There are two extreme answers:

**Monte Carlo (λ=1):** Add up all the actual imagined rewards.
$$G_t^{\text{MC}} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots + \gamma^{H-1} r_{H-1}$$
- ✅ **Unbiased** — uses real (imagined) rewards
- ❌ **High variance** — small changes in early actions cause wildly different futures

**TD(0) (λ=0):** Just use the immediate reward + the value function's estimate.
$$G_t^{\text{TD}} = r_t + \gamma \cdot v(z_{t+1})$$
- ✅ **Low variance** — one-step estimate is stable
- ❌ **High bias** — if $v$ is wrong, everything is wrong

### The Solution: TD(λ) with λ=0.95

λ-returns mix both approaches:

$$V_\lambda(t) = r_t + \gamma\left[(1-\lambda) \cdot v(z_{t+1}) + \lambda \cdot V_\lambda(t+1)\right]$$

Computed backwards from step $H-1$ to step 0.

With $\lambda = 0.95$, we lean heavily toward Monte Carlo (use the imagined rewards) but still allow the value function to stabilise things. The 5% TD component acts as a "safety net" against high variance.

### Why Backwards Computation?

The recursion goes **right to left**. Starting from the last imagined step:

```
V_λ(H-1) = r_{H-1} + γ · v(z_H)                          ← pure bootstrap
V_λ(H-2) = r_{H-2} + γ · [(1-λ)·v(z_{H-1}) + λ·V_λ(H-1)]
...
V_λ(0)   = r_0 + γ · [(1-λ)·v(z_1) + λ·V_λ(1)]
```

Each step uses the result from the step above, like a chain of dominoes falling backwards.

---

## 12. The Three Training Phases

Dreamer alternates between data collection and three distinct training phases. This separation is crucial — mixing them would create circular gradient dependencies.

### Phase 1: World Model Update

**Goal:** Make the world model accurate.

Given a batch of real sequences (B=50, T=50) from the replay buffer:

1. **Encode** all observations → embeddings
2. **RSSM observe** — run the sequence through the posterior (using real observations)
3. **Reconstruction loss** — can the decoder recreate the observations from the latent state?
4. **Reward loss** — can the reward model predict the actual rewards from the latent state?
5. **KL loss** — does the prior (dynamics-only prediction) match the posterior (observation-informed)?

$$\mathcal{L}_{\text{model}} = \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{reward}}$$

All five components (encoder, GRU, prior heads, posterior heads, decoder, reward MLP) are updated together with one optimizer.

### Phase 2: Actor Update

**Goal:** Learn actions that maximise expected reward in imagination.

1. **Detach** all posterior states from Phase 1 (no gradient flows back to the world model)
2. **Imagine** H=15 steps from each starting state using the actor + prior dynamics
3. **Compute** imagined rewards and values
4. **λ-returns** as training signal
5. **Loss** = $-\text{mean}(\gamma^t \cdot V_\lambda(t))$

The actor optimizer updates only actor weights.

### Phase 3: Critic Update

**Goal:** Learn to accurately estimate cumulative reward.

1. **Detach** imagined features and λ-returns from Phase 2
2. **Predict** values at each imagined state
3. **Loss** = MSE between predicted values and λ-return targets

The value optimizer updates only value network weights.

### Why Separate Optimizers?

Each has different learning rates:
- Model: `6e-4` (fastest — needs to track changing data distribution)
- Actor: `8e-5` (slow — stability matters more than speed)  
- Value: `8e-5` (slow — a noisy value estimate destabilises the actor)

And different gradient detachment points prevent unhealthy interactions (e.g., the actor shouldn't change the world model to make imagined futures look good).

---

## 13. Exploration Noise

During data collection, we add Gaussian noise to the actor's output:

```python
action = actor(features) + 0.3 * randn()
action = clamp(action, -1, 1)
```

$\sigma = 0.3$ is a moderate amount of exploration. Without it, the actor would only visit states it already thinks are good (exploitation), and might miss better strategies (exploration).

This is simpler than the ε-greedy approach used in discrete RL. For continuous actions, additive Gaussian noise is the standard approach (also used in DDPG, TD3, etc.).

---

## 14. Action Repeat

Each action is repeated 2 times in the environment before the agent makes a new decision.

**Why?**
1. **Temporal correlation:** Consecutive frames are nearly identical. Giving the agent 2 frames per decision is wasteful — one is enough.
2. **Effective horizon:** With repeat=2, the agent's 1000-step trajectory covers 2000 real env steps — twice the effective horizon.
3. **Speed:** Fewer decision points = fewer neural network forward passes = faster data collection.

The rewards from both repeated steps are summed, so no reward signal is lost.

---

## 15. Tanh-Squashed Normal — Why Not Just a Normal Distribution?

The action space is bounded: $a \in [-1, 1]^3$ (steering, gas, brake).

A vanilla Normal distribution outputs values in $(-\infty, +\infty)$. We could clip, but clipping creates a "wall" that produces zero gradients at the boundary — the policy can't learn to move away from the edge.

**Solution:** Sample from a Normal, then squash through $\tanh$:

$$a = \tanh(x), \quad x \sim \mathcal{N}(\mu, \sigma)$$

Properties:
- **Bounded:** $\tanh$ maps $\mathbb{R} \to (-1, 1)$ — always valid
- **Smooth gradients:** No hard clipping, gradients flow smoothly
- **The distribution "piles up" near ±1:** When $\mu$ is large, most samples land near $+1$, giving a "confident" action

### Mean and Std Parameterisation

```python
mean = 5.0 * tanh(raw_mean / 5.0)      # bounded in [-5, 5] (pre-tanh space)
std  = softplus(raw_std + 1.6) + 1e-4   # starts at ~5.0, always positive
```

- `mean_scale = 5.0`: Caps the pre-tanh mean so the distribution stays well-behaved
- `init_std = 5.0` ($\text{raw\_init\_std} = \log(e^5 - 1) \approx 1.6$): Extremely wide initial distribution → maximum exploration at the start of training
- `min_std = 1e-4`: Prevents the distribution from collapsing to a point (numerical stability)

---

## 16. The Reparameterisation Trick — Backprop Through Randomness

The core insight that makes Dreamer work: **we need gradients to flow through random samples**.

### The Problem

In Phase 2, the actor generates an action by *sampling*:
```python
action = Normal(μ, σ).sample()  # random!
```

Normally, you can't backpropagate through a random operation. The gradient $\frac{\partial \text{loss}}{\partial \mu}$ doesn't exist because `sample()` is not a deterministic function of $\mu$.

### The Trick

Rewrite the sampling as:
$$a = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

Now $a$ is a *deterministic, differentiable function* of $\mu$ and $\sigma$ (plus a fixed random number $\epsilon$). We can compute:
$$\frac{\partial a}{\partial \mu} = 1, \quad \frac{\partial a}{\partial \sigma} = \epsilon$$

In PyTorch, this is `dist.rsample()` (the "r" stands for "reparameterised").

### Where It's Used in Dreamer

1. **RSSM stochastic state:** `stoch = mean + std * torch.randn_like(std)` — gradients flow through the latent state during world model training
2. **Actor actions:** `actor.get_dist(feat).rsample()` — gradients flow from imagined rewards back through actions into actor weights
3. **The entire imagination chain:** state₀ → action₀ → state₁ → action₁ → ... → reward_H — all reparameterised, all differentiable

This is what makes Dreamer fundamentally different from policy gradient methods: instead of estimating gradients via REINFORCE (high variance), it computes exact gradients through the model (low variance, but requires a differentiable model).

---

## 17. Episode-Based Replay Buffer

### Why Episode-Based (Not Transition-Based)?

A typical RL replay buffer stores individual $(s, a, r, s')$ transitions. Dreamer stores **whole episodes** instead. Why?

The RSSM is recurrent — it needs *temporal context*. You can't just feed it a random isolated frame; you need a sequence of consecutive observations so the GRU can build up its hidden state.

When sampling, we draw random **contiguous sub-sequences** of length T=50 from random episodes. This ensures temporal consistency — the GRU sees a coherent sequence of events, not a shuffled mess.

### The Capacity

`max_episodes = 1000` — the buffer holds the last 1000 complete episodes. With CarRacing episodes lasting ~500 steps (×2 action repeat = ~250 agent steps), that's roughly 250,000 transitions.

---

## 18. Putting It All Together — The Full Algorithm

```
INITIALISE world model (encoder, RSSM, decoder, reward model), actor, critic
PREFILL replay buffer with 5000 random-action steps

REPEAT until total_steps reached:

    ┌─── COLLECT DATA (1000 steps) ──────────────────────────────────┐
    │ For each step:                                                  │
    │   1. Encode current observation                                 │
    │   2. RSSM posterior update (belief + observation → state)       │
    │   3. Actor selects action + exploration noise                   │
    │   4. Environment step → next obs, reward, done                  │
    │   5. Store transition in episode buffer                         │
    │   6. If episode done: save episode, reset state                 │
    └─────────────────────────────────────────────────────────────────┘

    ┌─── TRAIN (100 gradient steps per cycle) ────────────────────────┐
    │ For each gradient step:                                         │
    │                                                                 │
    │   Sample batch: (B=50, T=50) sub-sequences from replay          │
    │                                                                 │
    │   PHASE 1 — WORLD MODEL                                        │
    │     · Encode observations → embeddings                          │
    │     · RSSM.observe → posterior states, prior states             │
    │     · Decode features → reconstructed images                    │
    │     · Predict rewards from features                             │
    │     · KL loss + reconstruction loss + reward loss               │
    │     · Update encoder, RSSM, decoder, reward model               │
    │                                                                 │
    │   PHASE 2 — ACTOR (in imagination)                              │
    │     · Start from detached posterior states                       │
    │     · Imagine H=15 steps (actor + RSSM prior)                   │
    │     · Compute imagined rewards and values                       │
    │     · λ-returns (backwards recursion)                           │
    │     · Actor loss = -mean(discounted λ-returns)                  │
    │     · Update actor                                              │
    │                                                                 │
    │   PHASE 3 — CRITIC                                              │
    │     · Predict values at imagined states (detached)              │
    │     · Value loss = MSE(predicted, λ-return targets)             │
    │     · Update value network                                      │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 19. Summary Table of Every Design Choice

| Design Choice | Value | Why |
|---|---|---|
| **RSSM (split state)** | $h_t$ (200D) + $s_t$ (30D) | Deterministic path for long-range memory, stochastic for uncertainty |
| **Prior** | $p(s_t \mid h_t)$ | Dynamics-only prediction; used during imagination |
| **Posterior** | $q(s_t \mid h_t, o_t)$ | Observation-informed belief; used during real-data training |
| **KL divergence** | $D_{\text{KL}}(q \| p)$ | Trains the prior to match the posterior → accurate dynamics |
| **Free nats** | 3.0 | Lets the posterior encode details the prior can't predict, prevents collapse |
| **Encoder** | 4 Conv layers → 1024D | Compresses 64×64×3 pixels into a compact vector |
| **Decoder** | 4 ConvTranspose layers | Forces the latent to encode enough info to reconstruct the image |
| **Reward model** | 2-layer MLP (400 units) | Predicts immediate reward from latent state |
| **Value model** | 3-layer MLP (400 units) | Estimates cumulative future reward (bootstrapping) |
| **Actor** | 4-layer MLP → TanhNormal | Produces bounded continuous actions with smooth gradients |
| **Imagination horizon** | H=15 | Long enough for meaningful planning, short enough to limit compounding errors |
| **λ-returns** | λ=0.95 | 95% Monte Carlo + 5% bootstrap — good bias-variance tradeoff |
| **Discount** | γ=0.99 | Values rewards ~100 steps into the future |
| **Action repeat** | 2 | Skip redundant frames, effectively double the planning horizon |
| **Exploration noise** | σ=0.3 Gaussian additive | Simple continuous exploration |
| **Reparameterisation** | `rsample()` everywhere | Enables exact gradients through stochastic nodes |
| **Batch size** | B=50, T=50 | Each gradient step sees 2500 state transitions |
| **Training schedule** | 1000 env steps → 100 grad steps | ~10:1 imagination-to-reality ratio |
| **Separate optimisers** | Model 6e-4, Actor/Value 8e-5 | Different learning rates + gradient isolation |
| **Grad clipping** | max_norm=100 | Prevents explosive gradients from BPTT through RSSM |
| **Observation normalisation** | pixels / 255 − 0.5 | Centers data around 0, matches original implementation |
| **Episode-based buffer** | 1000 episodes | RSSM needs contiguous temporal sequences |
| **Softplus + 0.1 for std** | $\sigma = \text{softplus}(\cdot) + 0.1$ | Ensures σ > 0.1, prevents posterior/prior collapse |

---

## Appendix: The Key Equations on One Page

**RSSM Dynamics (Prior):**
$$h_t = \text{GRU}(\text{ELU}(\text{fc}([s_{t-1}; a_{t-1}])),\; h_{t-1})$$
$$\mu_p, \sigma_p = \text{split}(\text{fc}(\text{ELU}(\text{fc}(h_t))))$$
$$s_t \sim \mathcal{N}(\mu_p,\; \text{softplus}(\sigma_p) + 0.1)$$

**RSSM Posterior:**
$$\mu_q, \sigma_q = \text{split}(\text{fc}(\text{ELU}(\text{fc}([h_t; e_t]))))$$
$$s_t \sim \mathcal{N}(\mu_q,\; \text{softplus}(\sigma_q) + 0.1)$$

**World Model Loss:**
$$\mathcal{L}_{\text{model}} = \underbrace{D_{\text{KL}}(q \| p)}_{\text{dynamics}} + \underbrace{\|o_t - \hat{o}_t\|^2}_{\text{reconstruction}} + \underbrace{\|r_t - \hat{r}_t\|^2}_{\text{reward}}$$

**λ-Returns (backward recursion):**
$$V_\lambda(t) = r_t + \gamma\big[(1-\lambda)\, v(z_{t+1}) + \lambda\, V_\lambda(t+1)\big]$$

**Actor Loss (maximise imagined returns):**
$$\mathcal{L}_{\text{actor}} = -\mathbb{E}\left[\sum_{t=0}^{H-2} \gamma^t \, V_\lambda(t)\right]$$

**Critic Loss (regress onto λ-returns):**
$$\mathcal{L}_{\text{value}} = \mathbb{E}\left[\|v(z_t) - \text{sg}(V_\lambda(t))\|^2\right]$$

where $\text{sg}(\cdot)$ = stop gradient (detach).
