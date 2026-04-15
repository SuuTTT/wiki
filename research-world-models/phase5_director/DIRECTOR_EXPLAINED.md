# Director: Every Design Choice Explained from Scratch

*Paper: "Deep Hierarchical Planning from Pixels" — Hafner et al., 2022*
*Reference code: `director/embodied/agents/director/`*

---

## Table of Contents

1. [The Big Picture: Why Do We Need Hierarchy?](#1-the-big-picture)
2. [The Core Insight: Dreamer's Limitation and Director's Fix](#2-the-core-insight)
3. [Manager and Worker — The Two-Level Hierarchy](#3-manager-and-worker)
4. [The Goal Autoencoder — Why Not Just Output a Goal Directly?](#4-the-goal-autoencoder)
5. [The Skill Space — Discrete Codes as the Manager's Language](#5-the-skill-space)
6. [Goal Reward — How the Worker Knows It's Getting Closer](#6-goal-reward)
7. [Temporal Abstraction — The Manager Thinks Slowly](#7-temporal-abstraction)
8. [The Training Loop — Joint Imagination with Two Timescales](#8-the-training-loop)
9. [Manager Delta — Relative vs Absolute Goals](#9-manager-delta)
10. [REINFORCE for the Manager, Backprop for the Worker — Why Different?](#10-reinforce-vs-backprop)
11. [Exploration Rewards — Disagreement and Adversarial Intrinsic Motivation](#11-exploration-rewards)
12. [The World Model — DreamerV2's Discrete RSSM Under the Hood](#12-the-world-model)
13. [Abstract Trajectories — How the Manager Sees the Future](#13-abstract-trajectories)
14. [Split Trajectories — How the Worker Sees the Future](#14-split-trajectories)
15. [The Slow Target Critic — Stability Through Inertia](#15-slow-target-critic)
16. [Reward Channels — How Three Signals Get Combined](#16-reward-channels)
17. [Decoded Goal Visualisation — What Does the Agent Dream About?](#17-decoded-goal-visualisation)
18. [Putting It All Together — The Full Algorithm](#18-putting-it-all-together)
19. [Summary Table of Every Design Choice](#19-summary-table)

---

## 1. The Big Picture

Imagine you want a robot to navigate a large maze from pixels. The robot gets a reward **only** when it reaches the exit — nothing before that. The maze has 100,000 steps per episode.

**DreamerV1/V2 approach:** Learn a world model, imagine 15 steps ahead, and train an actor to maximise reward in those 15 imagined steps.

**The problem:** 15 steps of imagination can see ~1 room ahead. The robot has no concept of "go to the hallway, then turn left to the kitchen, then exit through the back door." It can only make micro-decisions 15 steps at a time. With sparse reward, it almost never stumbles onto the exit by chance — so it never learns.

**The Director approach:** Split the brain in two:
- A **Manager** that thinks at a slow, abstract level: "head toward the kitchen area"
- A **Worker** that thinks fast and handles motor control: "turn left 3 degrees, accelerate by 0.2"

The Manager sets *latent goals* (abstract targets in the world model's internal space). The Worker receives dense reward for making progress toward those goals. Even if the environment gives no external reward for thousands of steps, the Worker always has a clear signal: "am I getting closer to the Manager's goal?"

This is **hierarchical reinforcement learning (HRL)** — but done entirely inside the imagination of a world model, which is what makes Director special.

---

## 2. The Core Insight: Dreamer's Limitation and Director's Fix

### Dreamer's Bottleneck

In DreamerV1/V2, the actor imagines $H = 15$ steps and maximises:

$$\mathcal{L}_{\text{actor}} = -\sum_{t=0}^{H-1} \gamma^t V_\lambda(t)$$

This works when:
- Rewards are **dense** (you get feedback every step, like CarRacing's track score)
- The **planning horizon** is sufficient (15 steps covers the relevant future)

This fails when:
- Rewards are **sparse** (nothing for 10,000 steps, then +1 at the goal)
- The task requires **long-horizon reasoning** (the exit is 500 steps away, but you can only see 15)

### Director's Solution

Don't try to solve the whole task in one flat imagination. Instead:

1. The **Manager** imagines at an *abstract timescale* — every $K=8$ steps, it picks a "direction" (a goal in latent space)
2. The **Worker** imagines at the *original timescale* — every step, it picks an action to move toward the goal
3. The Worker gets **dense reward** from the Manager's goal (cosine similarity in latent space)

The Manager navigates rooms. The Worker navigates within a room. Each operates at the level of difficulty it can handle.

```
Manager:  "Go toward the kitchen"  ───────────────►  "Now go toward the exit"
          (step 0)                    K=8 steps        (step 8)

Worker:   left,left,fwd,fwd,fwd,right,fwd,fwd,       right,fwd,fwd,...
          (step 0,1,2,3,4,5,6,7)                       (step 8,9,10,...)
```

---

## 3. Manager and Worker — The Two-Level Hierarchy

### The Worker

The Worker is almost identical to DreamerV1's actor-critic. It's an MLP that takes a **state** and outputs an **action distribution**. The key difference: its input includes a **goal**.

From the code ([hierarchy.py](director/embodied/agents/director/hierarchy.py#L92)):
```python
# Worker sees: current state + goal + delta (goal - state)
dist = self.worker.actor(sg({**latent, 'goal': goal, 'delta': delta}))
```

| Input | Shape | Meaning |
|---|---|---|
| `deter` | (1024,) | GRU deterministic state |
| `stoch` | (32×32,) | Discrete categorical stochastic state |
| `goal` | (1024,) | The Manager's target in latent space |
| `delta` | (1024,) | `goal - current_deter` — how far away is the goal? |

The Worker is trained to **reach the goal**. Its reward is the cosine similarity between its latent state and the goal (details in §6).

### The Manager

The Manager is a separate MLP actor-critic operating in a *different action space*. It doesn't output environment actions — it outputs **skills** (codes in the skill space, see §5).

From the config:
```yaml
skill_shape: [8, 8]         # Manager outputs an 8×8 discrete code
env_skill_duration: 8        # Manager updates goal every 8 env steps
```

The Manager's "action" is a skill code $z \in \{0,1\}^{8 \times 8}$ (one-hot per dimension). This skill is decoded into a latent goal via the Goal Autoencoder.

The Manager is trained to **maximise the extrinsic environment reward** (the sparse signal the environment provides) plus an exploration bonus.

### The Separation of Concerns

| Aspect | Manager | Worker |
|---|---|---|
| Action space | Skill codes $(8 \times 8)$ | Environment actions (e.g., continuous motor) |
| Update frequency | Every $K=8$ steps | Every step |
| Reward signal | Extrinsic environment reward | Intrinsic goal-reaching reward |
| Loss gradient | REINFORCE (discrete skills → not differentiable) | Backprop (continuous actions → differentiable) |
| What it learns | *Where* to go | *How* to get there |

---

## 4. The Goal Autoencoder — Why Not Just Output a Goal Directly?

### The Naive Approach (and Why It Fails)

You might think: "Just have the Manager output a 1024-dim latent vector as the goal." This doesn't work for three reasons:

1. **Most of the 1024-dim latent space is empty.** The RSSM only produces latent states that correspond to real observations. Random points in 1024-dim space are almost certainly *not* valid world states. The Manager would propose impossible goals.

2. **No gradient signal for the Manager.** If the Manager's action is a continuous 1024-dim vector and the Worker uses REINFORCE-style training, the Manager never gets useful gradient feedback — the action space is too large.

3. **No structure.** A raw 1024-dim vector has no compositional or discrete structure. The Manager can't express "roughly north" vs "roughly south" — everything is a unique opaque blob.

### The Goal Autoencoder Solution

Director introduces a **bottleneck** between the Manager's intent and the Worker's goal:

```
Manager's skill code (8×8 one-hot)
        │
        ▼
   ┌─────────┐
   │ Decoder  │  (MLP: skill → latent goal vector)
   └─────────┘
        │
        ▼
  Goal (1024-dim latent vector)
        │
        ▼
     Worker
```

And a matching encoder for training:

```
  Latent state (1024-dim)
        │
        ▼
   ┌─────────┐
   │ Encoder  │  (MLP: latent → skill code distribution)
   └─────────┘
        │
        ▼
  Skill code (8×8 one-hot)
```

This is a **VAE** (Variational Autoencoder) over the latent goal space. It's trained with:

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L233-L247):
```python
# VAE training on replay data
enc = self.enc({'goal': goal, 'context': context})
dec = self.dec({'skill': enc.sample(), 'context': context})
rec = -dec.log_prob(tf.stop_gradient(goal))   # Reconstruction loss
kl = tfd.kl_divergence(enc, self.prior)        # KL to uniform prior
loss = (rec + kl).mean()
```

$$\mathcal{L}_{\text{VAE}} = \underbrace{-\log p(\text{goal} \mid z)}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}(q(z|\text{goal}) \| p(z))}_{\text{regularisation}}$$

### Why This Works

1. **Valid goals only.** The decoder is trained on real latent states, so it can only produce outputs that look like real states. The Manager can't propose impossible goals.

2. **Discrete bottleneck.** The 8×8 one-hot code has $8^8 \approx 16.7$ million possible values but only 64 "active bits." This forces compression — the Manager must express its intent using a compact vocabulary.

3. **Gradient-friendly for Manager.** The Manager's action is a discrete categorical distribution. Even though it's non-differentiable (requiring REINFORCE), the search space is vastly smaller than 1024 continuous dimensions.

4. **KL regularisation** ensures the skill codes are spread out and use the full capacity of the code space, preventing mode collapse.

---

## 5. The Skill Space — Discrete Codes as the Manager's Language

### The Structure: 8 Groups of 8 Choices

```yaml
skill_shape: [8, 8]   # 8 categorical variables, each with 8 classes
```

Think of it as 8 "dial knobs," each with 8 positions. Each configuration says something like:

```
Knob 1: direction      → position 3 (northwest)
Knob 2: speed          → position 7 (fast)
Knob 3: body posture   → position 1 (upright)
...
Knob 8: some other feature → position 5
```

The agent doesn't know these meanings in advance — the Goal Autoencoder learns to assign meaning to each knob position through training.

### Why Discrete (One-Hot) Instead of Continuous?

From the config:
```yaml
goal_encoder:
  dist: onehot        # ← Encoder outputs discrete one-hot codes
  shape: [8, 8]
```

1. **REINFORCE works better with discrete actions.** Policy gradient methods estimate $\nabla_\theta \mathbb{E}[\text{reward}]$ by sampling. With 8 discrete choices per knob, each sample explores a meaningful alternative. With continuous actions, the random exploration is aimless.

2. **Avoids posterior collapse.** Continuous VAEs with powerful decoders often ignore the latent code entirely (the decoder learns to reconstruct from the context alone). Discrete codes force the decoder to actually use the information.

3. **Interpretable.** You can decode each skill code to see what goal it represents — the images in the paper's "Internal Goals" visualisation come from decoding skill codes back to pixels.

### The Prior

```python
# Uniform prior over skill codes
self.prior = tfutils.OneHotDist(tf.zeros(shape))  # uniform categorical
self.prior = tfd.Independent(self.prior, len(shape) - 1)
```

The prior is a uniform distribution over all skill codes. The KL term in the VAE loss pushes the encoder's output toward this uniform prior — ensuring all skill codes get used, not just a few.

---

## 6. Goal Reward — How the Worker Knows It's Getting Closer

The Worker's **intrinsic reward** measures how well the current latent state matches the Manager's goal. Director experiments with many similarity metrics. The default is `cosine_max`:

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L348-L352):
```python
elif self.config.goal_reward == 'cosine_max':
    gnorm = tf.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
    fnorm = tf.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
    norm = tf.maximum(gnorm, fnorm)
    return tf.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
```

### The Intuition

**Cosine similarity** measures the *direction* of two vectors while normalising for magnitude:

$$r_{\text{goal}}(t) = \frac{\text{goal} \cdot \text{feat}_t}{\max(\|\text{goal}\|, \|\text{feat}_t\|)^2}$$

Wait — why `max(gnorm, fnorm)` instead of just `gnorm * fnorm` like standard cosine similarity?

**Standard cosine similarity** has a problem: if the feature vector `feat` has a very small norm (e.g., the agent is in a "boring" state), the similarity can still be +1 just because the direction happens to match. The reward signal is misleadingly strong.

**`cosine_max`** divides by the *larger* norm. This means:
- If `feat` points in the right direction AND has a similar magnitude to `goal` → high reward
- If `feat` points in the right direction but is much smaller → the reward is reduced
- This prevents the agent from getting "free reward" by collapsing its state to a tiny vector that happens to point the right way

### Why Not Euclidean Distance?

The paper tries `norm` (negative L2 distance) and `squared` (negative L2 squared), but they perform worse. The issue: Euclidean distance treats a state 10 units north of the goal the same as 10 units east. Cosine similarity is more robust because it separates *direction* (am I heading toward the goal?) from *progress* (how far have I gone?).

### The Full Menu of Reward Functions

The code contains 15+ options. Here are the key ones:

| Name | Formula | When to use |
|---|---|---|
| `cosine_max` | $\frac{g \cdot f}{\max(\|g\|, \|f\|)^2}$ | Default — robust and well-behaved |
| `normed_inner` | $\frac{g}{\|g\|} \cdot \frac{f}{\|f\|}$ | Pure direction matching |
| `norm` | $-\|g - f\|$ | L2 distance (less robust) |
| `squared` | $-\|g - f\|^2$ | Squared L2 (strong penalty for large errors) |
| `diff` | $\hat{g} \cdot (f_{t+1} - f_t)$ | Reward for *movement toward goal* per step |
| `enclogprob` | $\log q(z_{\text{skill}} \mid f_t)$ | Probability that state encodes the target skill |

---

## 7. Temporal Abstraction — The Manager Thinks Slowly

### The Timing

```yaml
env_skill_duration: 8       # Manager picks a new goal every 8 env steps
train_skill_duration: 8     # Same during imagination training
```

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L82-L88):
```python
update = (carry['step'] % duration) == 0
switch = lambda x, y: (
    tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
    tf.einsum('i,i...->i...', update.astype(x.dtype), y))
skill = sg(switch(carry['skill'], self.manager.actor(sg(latent)).sample()))
```

This code implements: "If step count is divisible by $K$ → Manager picks a new skill. Otherwise, keep the current skill."

### Why $K=8$?

This is a hyperparameter that balances two tensions:

| K too small | K too large |
|---|---|
| Manager changes goals so rapidly it's effectively a flat policy. No temporal abstraction benefit. | Goals stay fixed too long. The Worker might reach the goal early and have nothing to do. Or the goal becomes obsolete as the world changes. |

$K=8$ was found empirically to balance these forces. With an imagination horizon of $H=16$, the Manager gets to see $16/8 = 2$ decision points during training — enough to learn a meaningful sequence of goals.

### The Carry State

The `policy` function maintains a **carry state** across steps:

```python
carry = {'step': carry['step'] + 1, 'skill': skill, 'goal': goal}
```

- `step`: Integer counter that ticks up every imagination step
- `skill`: The current 8×8 skill code (held fixed between Manager updates)
- `goal`: The decoded 1024-dim goal vector (held fixed along with the skill)

This is different from Dreamer, where the policy is stateless (memoryless Markov) — here, the policy is **non-Markovian** because it remembers its current committed goal.

---

## 8. The Training Loop — Joint Imagination with Two Timescales

### The "Jointly New" Training Mode

From the config:
```yaml
jointly: new    # Train Manager and Worker jointly in one imagination trace
```

This is the most important training mode. Here's what happens:

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L131-L148):
```python
def train_jointly(self, imagine, start):
    with tf.GradientTape(persistent=True) as tape:
        # 1. Imagine a full trajectory using BOTH Manager and Worker
        policy = functools.partial(self.policy, imag=True)
        traj = self.wm.imagine_carry(
            policy, start, self.config.imag_horizon,
            self.initial(len(start['is_first'])))
        # 2. Compute all three reward channels
        traj['reward_extr'] = self.extr_reward(traj)  # environment reward
        traj['reward_expl'] = self.expl_reward(traj)  # exploration bonus
        traj['reward_goal'] = self.goal_reward(traj)  # Worker's goal reward
        traj['delta'] = traj['goal'] - self.feat(traj)
        # 3. Slice the trajectory differently for Worker and Manager
        wtraj = self.split_traj(traj)     # Worker sees fine-grained steps
        mtraj = self.abstract_traj(traj)  # Manager sees coarse abstract steps
    # 4. Update both
    mets = self.worker.update(wtraj, tape)
    mets = self.manager.update(mtraj, tape)
```

### Step by Step

**Step 1 — Imagine:** The world model runs a full $H=16$ step imagination. At every step, the `policy` function is called, which internally:
- Checks if it's time for the Manager to decide (every $K=8$ steps)
- If yes: Manager samples a skill → decode to goal
- If no: keep the existing goal
- Worker takes an action conditioned on the goal

The imagination produces a trajectory with: states, actions, skills, goals, rewards, continuation flags.

**Step 2 — Compute rewards:** Three types are computed over the trajectory:
- `reward_extr`: from the world model's reward head (the task signal)
- `reward_expl`: from the exploration module (curiosity)
- `reward_goal`: cosine similarity between state and goal (Worker's training signal)

**Step 3 — Slice differently:** This is the critical insight:

The **Worker** sees the trajectory sliced into chunks of $K$ steps (**"split trajectories"** — see §14). Each chunk is a mini-episode where the goal is fixed.

The **Manager** sees the trajectory **abstracted** to one data-point per $K$ steps (**"abstract trajectories"** — see §13). It processes a shorter, coarser-grained sequence.

**Step 4 — Update:** Each agent runs its own actor-critic update on its view of the trajectory.

---

## 9. Manager Delta — Relative vs Absolute Goals

```yaml
manager_delta: False   # Default: absolute goals
```

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L90-L93):
```python
new_goal = self.dec({'skill': skill, 'context': self.feat(latent)}).mode()
new_goal = (
    self.feat(latent).astype(tf.float32) + new_goal
    if self.config.manager_delta else new_goal)
```

### Absolute Goals (default)

The decoded goal is used directly as the target state. The Manager says: "The Worker should end up in a state that looks like *this*."

### Relative Goals (delta mode)

The decoded goal is *added* to the current state. The Manager says: "The Worker should move *this much* in latent space from where it is now."

### When to Use Which

| Mode | Advantage | Disadvantage |
|---|---|---|
| **Absolute** | Manager can target any known region of latent space. Good for navigation ("go to the kitchen"). | Goal might be far from current state — infeasible for Worker. |
| **Relative (delta)** | Always proposes nearby feasible goals. Scale-invariant. | Can't express "go back to the start" if that means a large state change. |

The paper finds absolute goals work better in most settings, so `manager_delta: False` is the default.

---

## 10. REINFORCE for the Manager, Backprop for the Worker — Why Different?

### The Worker Uses Backprop

The Worker outputs **continuous actions** (e.g., motor torques). The entire chain is differentiable:

```
Worker(state, goal) → action → RSSM.img_step(state, action) → next_state → reward_model(next_state) → reward
```

We can compute exact gradients $\frac{\partial \text{reward}}{\partial \theta_{\text{worker}}}$ via backpropagation through the world model. This is **low variance** — exactly the same trick as DreamerV1.

From the config:
```yaml
actor_grad_cont: backprop   # For continuous actions: straight-through backprop
```

### The Manager Uses REINFORCE

The Manager outputs **discrete skill codes** (8×8 one-hot). Sampling from a discrete distribution is **not differentiable** — you can't backprop through `onehot_dist.sample()`.

```python
mconfig = config.update({
    'actor_grad_cont': 'reinforce',  # ← Override for Manager
})
```

So the Manager uses the **REINFORCE** estimator (a.k.a. score function estimator):

$$\nabla_\theta \mathbb{E}[R] = \mathbb{E}\left[R \cdot \nabla_\theta \log \pi(a \mid s)\right]$$

In code ([agent.py](director/embodied/agents/director/agent.py#L362-L367)):
```python
if self.grad == 'reinforce':
    loss = -policy.log_prob(action)[:-1] * tf.stop_gradient(score)
```

"Increase the log-probability of skills that led to high reward. Decrease for low reward."

### Why This Asymmetry?

| Method | Requires | Variance | When to use |
|---|---|---|---|
| **Backprop** | Differentiable path from action to reward | Very low | Continuous actions through differentiable world model |
| **REINFORCE** | Only needs reward signal | High (needs more samples) | Discrete actions, or when the path is non-differentiable |

The Manager has no choice — its discrete outputs break the gradient chain. But the Worker can use backprop because its continuous actions flow smoothly through the RSSM.

### Entropy Regularisation

Both agents are entropy-regularised to maintain exploration:

```yaml
actent:
  impl: mult          # Multiplicative automatic tuning
  target: 0.5         # Target entropy (fraction of maximum)
manager_actent: 0.5   # Manager-specific entropy target
```

The system automatically adjusts the entropy coefficient so that each policy maintains ~50% of its maximum possible entropy. This prevents premature commitment to a single strategy.

---

## 11. Exploration Rewards — Disagreement and Adversarial Intrinsic Motivation

### The Problem

Sparse reward + hierarchy helps, but the Manager still needs to *discover* rewarding states before it can learn to navigate to them. Director uses **intrinsic motivation** to encourage systematic exploration.

### Exploration via Disagreement (`disag`)

From [expl.py](director/embodied/agents/director/expl.py#L9-L29):
```python
class Disag(tfutils.Module):
    def __call__(self, traj):
        inputs = self.inputs(traj)
        preds = [head(inputs).mode() for head in self.nets]
        disag = tf.math.reduce_std(preds, 0).mean(-1)
        return disag[1:]
```

Train an ensemble of $N=8$ prediction heads on the same data. The **disagreement** (standard deviation across heads) signals novelty:

- **High disagreement** → the model is uncertain → this region is underexplored → high exploration reward
- **Low disagreement** → the model confidently agrees → this region is well-understood → low exploration reward

This is the same idea as Random Network Distillation (RND) but using model ensemble disagreement instead.

### Adversarial Exploration (`adver`)

An alternative where the exploration reward is the VAE's **reconstruction error** — how surprised is the Goal Autoencoder by the current state?

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L398-L407):
```python
def elbo_reward(self, traj):
    enc = self.enc({'goal': feat, 'context': context})
    dec = self.dec({'skill': enc.sample(), 'context': context})
    # Reward = how hard it is to reconstruct the current state
    return ((dec.mode() - feat) ** 2).mean(-1)[1:]
```

States that are hard to reconstruct are *novel* — the Goal Autoencoder hasn't seen anything like them. This drives the agent to visit diverse states.

### How Exploration Integrates

From the config:
```yaml
manager_rews: {extr: 1.0, expl: 0.1, goal: 0.0}
worker_rews:  {extr: 0.0, expl: 0.0, goal: 1.0}
```

- The Manager receives: **1.0 × extrinsic + 0.1 × exploration** (task-focused, but with a curiosity nudge)
- The Worker receives: **1.0 × goal reaching** only (purely follows the Manager's commands)

This clean separation ensures the Worker is a reliable "executor" while exploration is the Manager's responsibility.

---

## 12. The World Model — DreamerV2's Discrete RSSM Under the Hood

Director uses the **DreamerV2-style discrete RSSM**, which is more powerful than DreamerV1's Gaussian RSSM.

### Discrete Stochastic State

Instead of DreamerV1's 30-dim Gaussian $s_t$, DreamerV2/Director uses:

```yaml
rssm: {deter: 1024, stoch: 32, classes: 32}
```

- **Deterministic state** $h_t$: 1024-dim GRU output (same concept as DreamerV1)
- **Stochastic state** $s_t$: 32 categorical variables, each with 32 classes

Total stochastic dimensions: $32 \times 32 = 1024$ one-hot values. When flattened, this is a sparse 1024-dim binary vector.

### Why Discrete?

DreamerV1 used Gaussian latents:
$$s_t \sim \mathcal{N}(\mu, \sigma)$$

DreamerV2/Director switched to categorical:
$$s_t \sim \text{Categorical}(\text{logit}_1), \ldots, \text{Categorical}(\text{logit}_{32})$$

Each of the 32 variables picks one of 32 classes via one-hot sampling.

**Benefits:**
1. **No variance collapse.** Gaussian posteriors can shrink their $\sigma$ to near-zero, killing the stochastic information path. Categorical distributions always sample discretely from their support — the information channel stays open.
2. **Better KL control.** The KL between two categoricals is bounded (by $\log K$), making optimisation more stable.
3. **Straight-through gradients.** During backprop, the one-hot sample is replaced by its soft probabilities (straight-through estimator), giving clean gradients.

### Custom GRU

The RSSM uses a custom GRU that's slightly different from the standard one:

From [nets.py](director/embodied/agents/director/nets.py#L136-L145):
```python
def _gru(self, x, deter):
    x = tf.concat([deter, x], -1)
    x = self.get('gru', Dense, units=3 * self._deter)(x)
    reset, cand, update = tf.split(x, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = tf.math.tanh(reset * cand)
    update = tf.nn.sigmoid(update - 1)       # ← biased toward remembering!
    deter = update * cand + (1 - update) * deter
    return deter, deter
```

The key trick: `sigmoid(update - 1)`. At initialisation, `update ≈ 0`, which means `deter ≈ (1 - 0) * old_deter` — the GRU **defaults to remembering** its previous state. This is the "forget bias" trick that helps RNNs preserve long-term memory.

### Balanced KL Loss

From [nets.py](director/embodied/agents/director/nets.py#L161-L165):
```python
def kl_loss(self, post, prior, balance=0.8):
    lhs = kl(stop_grad(post), prior)     # Train prior toward fixed post
    rhs = kl(post, stop_grad(prior))     # Train post toward fixed prior
    return balance * lhs + (1 - balance) * rhs
```

Instead of just $D_{\text{KL}}(q \| p)$ (DreamerV1 style), DreamerV2/Director uses a **balanced KL**:

$$\mathcal{L}_{\text{KL}} = 0.8 \cdot D_{\text{KL}}(\hat{q} \| p) + 0.2 \cdot D_{\text{KL}}(q \| \hat{p})$$

where $\hat{q}$ and $\hat{p}$ denote stop-gradient versions.

- The **first term** (weight 0.8): trains the prior to match the posterior. "Improve your dynamics prediction."
- The **second term** (weight 0.2): trains the posterior to match the prior. "Don't deviate too far from what's predictable."

This asymmetry means the dynamics model gets 4× more gradient pressure to learn good transitions, while the posterior has mild pressure to stay regular.

### Adaptive KL Scaling (AutoAdapt)

Instead of DreamerV1's fixed `free_nats = 3.0`, Director uses an **adaptive scaling** mechanism:

```yaml
wmkl: {impl: mult, scale: 0.1, target: 3.5, min: 1e-5, max: 1.0, vel: 0.1}
```

The system automatically adjusts the KL coefficient to maintain a target KL of ~3.5 nats. If KL drifts above 3.5, the coefficient increases to push it down. If below, the coefficient decreases to let the posterior use more information.

This is more robust than a fixed clamp because it adapts to different environments and training phases.

---

## 13. Abstract Trajectories — How the Manager Sees the Future

The Manager operates at a coarse timescale. A 16-step imagination trajectory needs to be **compressed** into 2 "Manager steps" ($16 / K = 16/8 = 2$).

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L431-L443):
```python
def abstract_traj(self, traj):
    traj['action'] = traj.pop('skill')   # Manager's "action" is the skill code
    k = self.config.train_skill_duration  # K=8
    reshape = lambda x: x.reshape([x.shape[0] // k, k] + x.shape[1:])
    weights = tf.math.cumprod(reshape(traj['cont'][:-1]), 1)
    for key, value in list(traj.items()):
        if 'reward' in key:
            # Average rewards over K steps, weighted by survival probability
            traj[key] = (reshape(value) * weights).mean(1)
        elif key == 'cont':
            # Continuation = product over K steps (did we survive the whole interval?)
            traj[key] = tf.concat([value[:1], reshape(value[1:]).prod(1)], 0)
        else:
            # States: take the first state of each K-step block
            traj[key] = tf.concat([reshape(value[:-1])[:, 0], value[-1:]], 0)
```

### What the Manager Sees

Given a 16-step trajectory indexed $t = 0, 1, \ldots, 16$:

| Original | steps 0-7 | steps 8-15 | step 16 |
|---|---|---|---|
| **Abstract state** | $s_0$ | $s_8$ | $s_{16}$ |
| **Abstract action** | skill at $t=0$ | skill at $t=8$ | — |
| **Abstract reward** | $\frac{1}{8}\sum_{t=0}^{7} r_t \cdot w_t$ | $\frac{1}{8}\sum_{t=8}^{15} r_t \cdot w_t$ | — |
| **Abstract cont** | $\prod_{t=1}^{8} c_t$ | $\prod_{t=9}^{16} c_t$ | — |

The rewards are **survival-weighted averages** — if the agent dies at step 5, only steps 0-4 contribute to the Manager's reward. The continuation flag is the *product* — the Manager survives a block only if the Worker survived every step within it.

### Why This Matters

The Manager's critic and actor operate on this abstract trajectory. They see a 2-step sequence (from 16 real steps) and maximize 2-step λ-returns. This is tractable even if the total imagination horizon grows much longer, because the Manager always sees $H/K$ steps.

---

## 14. Split Trajectories — How the Worker Sees the Future

The Worker needs to be trained on goal-conditioned episodes where the goal is **fixed within each segment**. A 16-step trajectory with 2 Manager decisions becomes 2 independent sub-episodes for the Worker.

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L418-L429):
```python
def split_traj(self, traj):
    k = self.config.train_skill_duration  # K=8
    # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
    # Note: boundaries overlap! State at step 4 is both the end of chunk 1
    # and the start of chunk 2.
    val = tf.concat([reshape(val[:-1]), val[k::k][:, None]], 1)
    # Transpose so k-step chunks become the "batch" dimension
    val = val.transpose([1, 0] + list(range(2, len(val.shape))))
    val = val.reshape([val.shape[0], np.prod(val.shape[1:3])] + val.shape[3:])
```

### What the Worker Sees

The 16-step trajectory `[s₀, s₁, ..., s₁₆]` is chopped into:

| Worker chunk 1 | Worker chunk 2 |
|---|---|
| $s_0 \to s_1 \to \ldots \to s_8$ | $s_8 \to s_9 \to \ldots \to s_{16}$ |
| Goal = $g_0$ (fixed from Manager at $t=0$) | Goal = $g_8$ (fixed from Manager at $t=8$) |

These chunks are **stacked into the batch dimension**. If the original batch had $N$ parallel trajectories, the Worker now sees $2N$ short episodes of length $K+1 = 9$.

The Worker's critic and actor operate on these short goal-conditioned episodes. The Worker sees dense reward every step (cosine similarity to goal) and learns to efficiently reach whatever goal it's given.

### The Boundary Trick

Notice the overlap at step 8: it's the last state of chunk 1 **and** the first state of chunk 2. Also:

```python
# Bootstrap sub trajectory against current not next goal.
traj['goal'] = tf.concat([traj['goal'][:-1], traj['goal'][:1]], 0)
```

The bootstrap value at the end of each chunk uses the *current* goal, not the next one. This prevents the Worker from being confused by goal changes at chunk boundaries.

---

## 15. The Slow Target Critic — Stability Through Inertia

Both the Manager's and Worker's critics use a **slow target network** for computing return targets:

From [agent.py](director/embodied/agents/director/agent.py#L426-L430):
```python
if self.config.slow_target:
    self.target_net = nets.MLP((), **self.config.critic)
    self.updates = tf.Variable(-1, dtype=tf.int64)
```

```yaml
slow_target: True
slow_target_update: 100     # Copy weights every 100 updates
slow_target_fraction: 1.0   # Full copy (not EMA)
```

### The Problem Without a Target Network

The critic estimates $V(s)$ and its own predictions are used to compute the training targets (via λ-returns). This creates a **circular dependency**: the critic is trained on targets that depend on its own values.

If the critic changes rapidly, these targets shift unpredictably, causing oscillation or divergence. This is the deadly "moving target" problem that plagued early value-based RL.

### The Solution

Maintain a **frozen copy** of the critic (the "target network"). Use this frozen copy to compute λ-return targets. Every 100 gradient steps, refresh the frozen copy with the current critic's weights.

This breaks the circular dependency: targets are stable between refreshes, giving the critic a fixed target to regress toward.

### Hard vs Soft Updates

Director uses `slow_target_fraction: 1.0` — a full copy every 100 steps (hard update, like DQN). An alternative is exponential moving average (EMA, like SAC):

$$\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{current}} + (1 - \tau) \cdot \theta_{\text{target}}$$

Hard updates are simpler and work well with the relatively slow imagination-based training.

---

## 16. Reward Channels — How Three Signals Get Combined

Each agent receives a **weighted mixture** of three reward types:

```yaml
manager_rews: {extr: 1.0, expl: 0.1, goal: 0.0}
worker_rews:  {extr: 0.0, expl: 0.0, goal: 1.0}
```

### Manager's Reward

$$r_{\text{manager}} = 1.0 \cdot r_{\text{extrinsic}} + 0.1 \cdot r_{\text{exploration}}$$

- **Extrinsic (1.0):** The real environment reward — the actual task signal
- **Exploration (0.1):** A small curiosity bonus to push the Manager into novel regions
- **Goal (0.0):** The Manager gets no credit for the Worker reaching goals — that would be circular

### Worker's Reward

$$r_{\text{worker}} = 1.0 \cdot r_{\text{goal}}$$

- **Extrinsic (0.0):** The Worker is completely agnostic to the task — it only follows orders
- **Exploration (0.0):** The Worker doesn't explore on its own
- **Goal (1.0):** 100% of the Worker's reward comes from reaching the Manager's goal

### Why This Separation?

This is the key architectural decision that makes HRL stable:

1. **The Manager discovers rewarding goals.** It gets sparse environment reward + curiosity, so it explores the world and learns which regions yield rewards.

2. **The Worker reliably executes goals.** It gets dense goal reward, so it can always learn — even when environment reward is zero.

3. **No signal mixing.** If the Worker also received extrinsic reward, it might learn to "cheat" — reaching high-reward states regardless of the Manager's commands, making the Manager irrelevant.

### Each Reward Gets Its Own Critic

Importantly, each reward channel has a **separate V-function** (critic):

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L25-L32):
```python
self.worker = agent.ImagActorCritic({
    'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig),
    'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig),
    'goal': agent.VFunction(lambda s: s['reward_goal'], wconfig),
}, config.worker_rews, act_space, wconfig)
```

Even though the Worker only uses `goal`, all three critics are defined (but only active ones with non-zero scale actually train). The scores from all active critics are combined before updating the actor.

---

## 17. Decoded Goal Visualisation — What Does the Agent Dream About?

The second attached image shows "Environment" vs "Internal Goals" for four different tasks. The Internal Goals are **decoded back to pixel space** by running goal vectors through the world model's image decoder.

From [hierarchy.py](director/embodied/agents/director/hierarchy.py#L96-L99):
```python
if 'image' in self.wm.heads['decoder'].shapes:
    outs['log_goal'] = self.wm.heads['decoder']({
        'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal),
    })['image'].mode()
```

### How Goal Decoding Works

1. The goal is a 1024-dim deterministic latent vector
2. `get_stoch(goal)` computes the RSSM's prior stochastic state consistent with this deterministic state — i.e., "what stochastic details would this state likely have?"
3. The decoder takes `(deter=goal, stoch=predicted_stoch)` and produces a pixel reconstruction

### Why the Internal Goals Are Often Dark/Blurry

Looking at the visualisation: the internal goals appear mostly dark with faint structures. This is because:

1. **Goals are in `deter` space only.** The RSSM deterministic state captures the *average* appearance of a region (like a cluster centre). Fine details are in the stochastic part, which is inferred.

2. **The goal autoencoder discards detail.** The 8×8 discrete bottleneck forces lossy compression. The decoded goal only preserves the *gist* — "there's a room with an object in the corner" — not pixel-perfect reconstruction.

3. **This is by design.** The Worker doesn't need pixel-perfect goals. It needs a *direction* in latent space. The faint blurry image contains enough information about where the Manager wants the Worker to end up.

---

## 18. Putting It All Together — The Full Algorithm

```
INITIALISE world model (RSSM, encoder, decoder, reward head, cont head)
INITIALISE Manager (skill actor, critics for extr+expl+goal)
INITIALISE Worker (action actor, critics for extr+expl+goal)
INITIALISE Goal Autoencoder (encoder: state→skill, decoder: skill→goal)

REPEAT until done:

    ┌─── COLLECT DATA ───────────────────────────────────────────────┐
    │ step_counter = 0                                                │
    │ For each env step:                                              │
    │   Encode observation → latent state                             │
    │   If step_counter % K == 0:                                     │
    │       Manager samples skill from its actor                      │
    │       Decode skill → goal (1024-dim latent target)              │
    │   Worker selects action given (state, goal, delta)              │
    │   Execute action in environment                                 │
    │   Store (obs, action, reward, done) in replay buffer            │
    │   step_counter += 1                                             │
    └─────────────────────────────────────────────────────────────────┘

    ┌─── TRAIN WORLD MODEL ──────────────────────────────────────────┐
    │ Sample batch from replay buffer                                 │
    │ Run RSSM observe → posterior + prior states                     │
    │ Decode states → predicted observations + rewards + continuations│
    │ Loss = balanced_KL + reconstruction + reward + continuation     │
    │ Update encoder, RSSM, decoder, reward head, cont head           │
    └─────────────────────────────────────────────────────────────────┘

    ┌─── TRAIN GOAL AUTOENCODER ─────────────────────────────────────┐
    │ From replay: take pairs (context_state, target_state)           │
    │ Encode target → skill code distribution                         │
    │ Decode skill code → reconstructed target                        │
    │ Loss = reconstruction + KL(encoder ‖ uniform_prior)             │
    │ Update goal encoder + goal decoder                              │
    └─────────────────────────────────────────────────────────────────┘

    ┌─── TRAIN MANAGER + WORKER (Joint Imagination) ─────────────────┐
    │ Start from posterior states (real data anchor points)            │
    │                                                                 │
    │ IMAGINE H=16 steps:                                             │
    │   Every K=8 steps: Manager samples skill → decode → goal        │
    │   Every step: Worker(state, goal) → action → RSSM.img_step     │
    │   Compute: reward_extr, reward_expl, reward_goal at each step   │
    │                                                                 │
    │ SPLIT for Worker:                                               │
    │   Chop into K-step sub-episodes, each with fixed goal           │
    │   Worker reward = cosine_max(state, goal)                       │
    │   Compute λ-returns over each sub-episode                       │
    │   Actor loss = -weighted returns (backprop)                     │
    │   Critic loss = MSE between value prediction and λ-returns      │
    │                                                                 │
    │ ABSTRACT for Manager:                                           │
    │   Compress to 1 point per K steps (average rewards, take first  │
    │   state)                                                        │
    │   Manager reward = extrinsic + 0.1 × exploration                │
    │   Compute λ-returns over abstract trajectory                    │
    │   Actor loss = -REINFORCE(skill, score) + entropy reg           │
    │   Critic loss = -log_prob(target) on return targets             │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 19. Summary Table of Every Design Choice

| Design Choice | Value / Detail | Why |
|---|---|---|
| **Two-level hierarchy** | Manager + Worker | Separates *where to go* from *how to get there*. Enables long-horizon planning with dense worker reward. |
| **Skill space** | 8×8 one-hot categorical | Discrete bottleneck forces compact symbolic communication. REINFORCE-friendly action space. |
| **Goal Autoencoder** | VAE: state→skill→goal | Ensures goals are valid latent states. Regularised by KL to uniform prior. |
| **Goal = deterministic state** | 1024-dim `deter` only | The deterministic state captures the "where" — stochastic part is noise. Only needing `deter` simplifies the goal space. |
| **Manager update interval** | $K = 8$ steps | Balances temporal abstraction with responsiveness. |
| **Goal reward** | `cosine_max` (default) | Robust directional similarity. Dividing by `max(‖g‖, ‖f‖)` prevents "free reward" from tiny-norm states. |
| **Manager gradient** | REINFORCE | Discrete skill codes are non-differentiable — can't backprop through one-hot sampling. |
| **Worker gradient** | Backprop | Continuous actions through differentiable world model — exact gradients. |
| **Manager reward** | 1.0×extrinsic + 0.1×exploration | Task-focused with a curiosity bonus for discovering new rewarding regions. |
| **Worker reward** | 1.0×goal only | Pure goal-reaching. No task knowledge — reliably follows commands. |
| **Separate critics per reward channel** | VFunction for extr, expl, goal | Each reward signal gets accurate value estimation. Scores are combined at the actor level. |
| **Discrete RSSM** | 32 categoricals × 32 classes | No variance collapse. Bounded KL. Straight-through gradients. (DreamerV2 upgrade) |
| **Balanced KL** | 0.8×forward + 0.2×reverse | Trains dynamics model (prior) 4× harder than posterior regularisation. |
| **Adaptive KL scaling** | AutoAdapt, target=3.5 nats | Auto-tunes the KL coefficient instead of fixed free_nats. |
| **Custom GRU with forget bias** | `sigmoid(update - 1)` | Biases gate toward "remember previous state." Better long-term memory. |
| **Slow target critic** | Hard copy every 100 updates | Stabilises λ-return targets, preventing value oscillation. |
| **Abstract trajectories** | $H/K$ steps for Manager | Compress H=16 steps into 2 Manager decisions. Rewards averaged, cont multiplied. |
| **Split trajectories** | K-step chunks for Worker | Each chunk is a mini-episode with fixed goal. Overlapping boundaries. |
| **Imagination horizon** | $H = 16$ | Long enough for 2 Manager decisions ($16/8 = 2$). |
| **Entropy regularisation** | Auto-tuned, target=50% max | Prevents premature convergence. Separate targets for Manager and Worker. |
| **Manager delta mode** | Off by default | Absolute goals more expressive than relative for navigation tasks. |
| **Exploration method** | Disagreement (8 ensemble heads) | Standard deviation across predictions = novelty signal. Also supports adversarial (reconstruction error). |
| **Symlog critic** | `dist: symlog` in config | Handles extreme reward scales by predicting in symlog space. |
| **Return normalisation** | `retnorm: std, decay=0.999` | Normalises returns to unit std, preventing value scale issues across reward channels. |
| **Replay chunks** | 64 consecutive frames | Enough temporal context for the RSSM to build accurate beliefs. |
| **Priority replay** | Optional, using reward loss | Samples sequences where the reward model struggles, focusing learning effort. |

---

## Appendix A: Director vs Dreamer — What Changed?

| Component | DreamerV1 | Director |
|---|---|---|
| **Actor** | Single flat actor | Manager (slow, discrete skills) + Worker (fast, motor control) |
| **Goal conditioning** | None | Worker receives latent goal + delta from Manager |
| **Skill space** | N/A | 8×8 one-hot categorical via Goal Autoencoder |
| **RSSM** | Gaussian stochastic (30D) | Discrete categorical (32×32) |
| **KL handling** | Fixed free_nats=3.0 clamp | Adaptive auto-tuning to target KL |
| **KL computation** | $D_{\text{KL}}(q \| p)$ | Balanced: $0.8 D_{\text{KL}}(\hat{q}\|p) + 0.2 D_{\text{KL}}(q\|\hat{p})$ |
| **Imagination** | 15 steps, flat | 16 steps, Manager decides every 8, Worker every 1 |
| **Actor gradient** | Backprop only | Manager: REINFORCE, Worker: Backprop |
| **Exploration** | Additive Gaussian on actions | Ensemble disagreement or adversarial reconstruction |
| **Reward model** | MSE | Symlog distribution |
| **Critic** | MSE value | Symlog-distributed value with slow target network |

## Appendix B: The Key Equations on One Page

**Goal Autoencoder (VAE):**
$$q(z \mid g, c) = \text{Encoder}(g, c) \qquad p(g \mid z, c) = \text{Decoder}(z, c)$$
$$\mathcal{L}_{\text{VAE}} = -\log p(g \mid z, c) + \alpha \cdot D_{\text{KL}}(q(z|g,c) \| \text{Uniform})$$

**Manager policy (every K steps):**
$$z_{\text{skill}} \sim \pi_{\text{mgr}}(s_t), \quad g_t = \text{Decoder}(z_{\text{skill}}, h_t)$$

**Worker policy (every step):**
$$a_t \sim \pi_{\text{wkr}}(s_t, g_t, g_t - h_t)$$

**Worker's intrinsic reward:**
$$r_{\text{goal}}(t) = \text{cosine\_max}(g, f_t) = \frac{g \cdot f_t}{\max(\|g\|, \|f_t\|)^2}$$

**Manager actor loss (REINFORCE):**
$$\mathcal{L}_{\text{mgr}} = -\sum_t \log \pi_{\text{mgr}}(z_t | s_t) \cdot \text{sg}(A_t) + \mathcal{L}_{\text{entropy}}$$

**Worker actor loss (backprop):**
$$\mathcal{L}_{\text{wkr}} = -\sum_t w_t \cdot V_\lambda^{\text{goal}}(t) + \mathcal{L}_{\text{entropy}}$$

**Critic loss (both agents):**
$$\mathcal{L}_{\text{critic}} = -\sum_t w_t \cdot \log p_\theta(V_\lambda(t) \mid s_t)$$

**Balanced KL (world model):**
$$\mathcal{L}_{\text{KL}} = 0.8 \cdot D_{\text{KL}}(\text{sg}(q) \| p) + 0.2 \cdot D_{\text{KL}}(q \| \text{sg}(p))$$

where $\text{sg}(\cdot)$ = stop gradient.
