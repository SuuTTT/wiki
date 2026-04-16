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

### Prompt 1: Explain an Algorithm Before Implementing

Use this BEFORE writing code to build understanding.

```
I'm about to implement {ALGORITHM} ({paper title, year}).

Before writing code, explain every design choice and trick in the paper
using the Feynman method — as if teaching someone who understands basic
neural networks and RL.

Target reader: Explicitly declare the required preliminaries at the beginning. Do NOT assume the reader knows advanced algorithms like DQN, SAC, or PPO unless explicitly stated as a prerequisite. State exactly what math (e.g., high school calculus, basic probability) or ML concepts (e.g., MLPs, CNNs) are needed, and what previous concepts this tutorial builds upon.

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

### Prompt 2: Small CleanRL Algorithm (DQN, PPO, SAC, etc.)

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

### Prompt 3: Large World Model Project (Dreamer, Director)

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
if __name__ == "__main__":
    # Test all shapes with dummy data
    batch, seq_len = 4, 50
    obs = torch.randn(batch, seq_len, *obs_shape)
    actions = torch.randn(batch, seq_len, act_dim)
    # ... test observe, imagine, decode, reward, continue
    print("ALL SHAPE TESTS PASSED")

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

### Prompt 4: Adding Hierarchy (Director, Options, HIRO)

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
{paste the 20-50 lines most likely to contain the bug}

I suspect the issue might be in: {your hypothesis}

Please:
1. Diagnose the most likely root cause
2. Show the exact fix (minimal diff)
3. Explain why this bug produces the observed symptom
```

---

### Prompt 6: Generate an EXPLAINED.md Tutorial (Distill-Style Blog Post)

Use this AFTER you understand the algorithm (Prompt 1) and have working code (Prompts 2-4).
This produces a publishable, journal-quality deep dive.

```
I have a working implementation of {ALGORITHM} in {FILE} and a reference
codebase at {REFERENCE_CODE_PATH}.

Create a comprehensive tutorial called {ALGORITHM}_EXPLAINED.md in the style
of Distill.pub / Colah's blog / Lilian Weng.

Target reader: Explicitly declare the required preliminaries at the beginning. Do NOT assume the reader knows advanced algorithms like DQN, SAC, or PPO unless explicitly stated as a prerequisite. State exactly what math (e.g., high school calculus, basic probability) or ML concepts (e.g., MLPs, CNNs) are needed, and what previous concepts this tutorial builds upon.

Structure:
- 15-20 numbered sections, each covering ONE design choice or component
- Start with "The Big Picture" (what problem, why existing methods fail)
- End with "Putting It All Together" (full algorithm pseudocode) and
  "Summary Table" (every design choice → value → why)

For EACH section:
1. Motivation: What goes wrong WITHOUT this? (concrete failure example)
2. Intuition: Explain like I'm smart but unfamiliar (analogies welcome)
3. Math: The exact equation in KaTeX, with every symbol defined
4. Code: Reference the specific file + line from {REFERENCE_CODE_PATH}
5. Hyperparameters: typical values and what happens if you change them

Style guide:
- Feynman method: "If you can't explain it simply, you don't understand it"
- Use concrete examples, not abstract definitions
- Tables for comparisons (this vs that, with vs without)
- ASCII diagrams for architecture flow
- First-person ("Why do WE need...") not third-person academic style
- No filler paragraphs — every sentence should teach something

Include:
- Appendix A: comparison table vs the predecessor algorithm
- Appendix B: all key equations on one page

Do NOT include: introduction/conclusion boilerplate, related work survey,
future work speculation, or acknowledgements.

Reference papers: {list paper titles + years}
Reference code: {path}
```

**Example filled in:**

```
I have a working implementation of DreamerV1 in
wiki/research-world-models/phase4_dreamer/04_dreamer_claude.py and a reference
codebase at cleanrl/cleanrl/.

Create a comprehensive tutorial called DREAMER_EXPLAINED.md in the style
of Distill.pub / Colah's blog / Lilian Weng.

Target reader: Explicitly declare the required preliminaries at the beginning. Do NOT assume the reader knows advanced algorithms like DQN, SAC, or PPO. State exactly what math (e.g., high school calculus, basic probability) or ML concepts (e.g., MLPs, CNNs) are needed, and what previous concepts this tutorial builds upon.

Structure:
- 19 numbered sections covering: world model, RSSM, prior/posterior,
  KL divergence, free nats, encoder/decoder, actor, critic, lambda returns,
  three training phases, exploration noise, action repeat, tanh-Normal,
  reparameterisation trick, replay buffer, full algorithm, summary table

For EACH section:
1. Motivation: What goes wrong WITHOUT this?
2. Intuition: Feynman-style explanation with analogies
3. Math: KaTeX equations with all symbols defined
4. Code: Reference specific lines from the codebase
5. Hyperparameters: typical values and sensitivity

Reference papers: "Dream to Control" (Hafner 2020), "World Models" (Ha 2018)
Reference code: wiki/research-world-models/phase4_dreamer/
```

---

### Prompt 7: Generate 3Blue1Brown-Style Video Script + Animation Spec

```
I have a technical blog post about {ALGORITHM} at {EXPLAINED_MD_PATH}.

Create a video production package for a 3Blue1Brown-style animated explainer:

PART A — SCRIPT (narration text, ~15 min when read aloud):
- Cold open: a compelling question or paradox (30 sec)
  e.g. "What if an RL agent could practice in its dreams?"
- Act 1: The problem (2 min) — why existing methods fail, with visual example
- Act 2: The key insight (3 min) — the "aha" moment, one core idea
- Act 3: How it works (7 min) — walk through components one by one
  Each component: show the equation building up term by term
- Act 4: The payoff (2 min) — results, what this enables
- Closing hook (30 sec) — tease the next video in the series

Style: conversational, precise, builds intuition before formalism.
NO "In this video we will..." or "Let's get started" filler.

PART B — ANIMATION SPEC (one per scene):
For each scene, specify:
- Scene number + timestamp range
- Narration text for that scene
- Visual description: what appears on screen
  - Use Manim-style descriptions: "A number line appears. A blue dot slides
    from 0 to 1 as we say 'probability'."
  - Specify: shapes, colors, arrows, text labels, transforms
  - For equations: which terms highlight, when terms appear/disappear
- Transition: how we move to the next scene (fade, zoom, slide)

PART C — MANIM SCENE LIST:
List each animation as a Manim class name + brief description:
  - `class ColdOpen(Scene)` — Agent in maze, reward appears far away
  - `class RSSMDiagram(Scene)` — RSSM architecture building up piece by piece
  - `class KLIntuition(Scene)` — Two distributions sliding toward each other
  etc.

Target: Someone with Manim experience can implement each scene independently.

Reference: {EXPLAINED_MD_PATH}
Algorithm: {ALGORITHM} ({paper title, year})
```

#### Manim Production Workflow

```
1. Generate script + animation spec (Prompt 7)
2. Review script for accuracy (you — check against EXPLAINED.md)
3. Generate Manim code scene by scene (use LLM, one scene per prompt)
4. Render scenes: manim -pql scene_file.py SceneName
5. Preview and iterate (adjust timing, colors, layout)
6. Stitch scenes: ffmpeg or Manim's built-in sequencing
7. Record voiceover (your voice, or TTS like ElevenLabs/Kokoro)
8. Merge audio + video: ffmpeg -i video.mp4 -i audio.mp3 -c:v copy final.mp4
9. Add subtitle track (.srt file, optional)
10. Upload to YouTube with thumbnail, description, timestamps
```

#### Manim Setup

```bash
pip install manim
# For LaTeX rendering (equations):
apt-get install texlive-full  # or texlive-latex-extra for lighter install
```

---

### Prompt 8: Generate TikTok / Short-Form Content Script

```
I have a technical blog post about {ALGORITHM} at {EXPLAINED_MD_PATH}.

Create 3 short-form video scripts (30-60 seconds each) for TikTok/Reels/Shorts.

SCRIPT FORMAT (for each):
- Hook (first 2 seconds): the text overlay + first words spoken
  Must stop the scroll. Use: a surprising claim, a visual paradox,
  or a "did you know" pattern.
- Body (20-40 seconds): ONE insight from the algorithm, explained
  with a visual metaphor that works WITHOUT sound
- Payoff (5-10 seconds): the "so what" — what this enables or why it matters
- CTA: "Follow for more ML explained simply" or similar

STYLE GUIDE:
- Talk like you're explaining to a smart friend at a bar, not lecturing
- Use analogies from everyday life (NOT other ML papers)
- Maximum one equation on screen (if any) — make it big and annotated
- Screen text should be readable in 2 seconds
- If using meme format, specify the template + your text overlay

FOR EACH SCRIPT, PROVIDE:
1. Hook text (on-screen) + spoken words
2. Visual description per beat (what's on screen each second)
3. Suggested audio: trending sound, your voiceover, or text-to-speech
4. Hashtags (5-8, mix of niche #ReinforcementLearning and broad #AI #Tech)

SCRIPT IDEAS to consider:
- "Why can't robots plan ahead?" → lead into world models
- "This AI literally dreams to learn" → Dreamer
- "Your brain has a Manager and a Worker, so does this AI" → Director
- "What if reward is too rare to learn from?" → sparse reward problem
- Meme: [Drake meme] "Training on real data" vs "Training in your dreams"

Reference: {EXPLAINED_MD_PATH}
Algorithm: {ALGORITHM}
```

#### Short-Form Production Workflow

```
1. Generate scripts (Prompt 8) — pick the strongest hook
2. Record or generate voiceover (30-60 sec)
3. Visual options (pick one):
   a. Screen recording of Manim animation (reuse from Tier 2)
   b. Slides with big text + diagram (Canva, Figma, Keynote)
   c. You talking to camera with B-roll of visualisations
   d. Meme template + text overlay (CapCut, InShot)
4. Edit in CapCut / DaVinci Resolve / iMovie
   - Add captions (auto-caption or manual .srt)
   - Add sound/music (trending audio or lo-fi background)
   - Cut dead air ruthlessly — every second must earn attention
5. Export: 1080x1920 (9:16), <60 sec, <100MB
6. Post with hashtags during peak hours (varies by platform)
```

---

### Prompt 9: Research Proposal (Before Implementing Your Idea)

Use this BEFORE Prompt 10. Write the proposal, circulate it, get feedback, THEN implement.

**Why this phase exists:**

Your supervisor is correct — and this is one of the highest-leverage habits in research.
The reason is simple: **implementation is expensive, thinking is cheap.**

A bad idea takes the same 3-7 days to implement as a good one. If you spend 2 days
coding, find it doesn't work, and only THEN realise the idea was flawed — that's 2 days
wasted. A 1-page proposal takes 2 hours. Showing it to 3 people takes 1 day. If someone
spots the flaw ("this was tried in 2019, see paper X" or "your assumption about X doesn't
hold because Y"), you saved those 2 days.

**The deeper reasons:**

1. **You don't know what you don't know.** You might be excited about an idea that was
   tried and abandoned 5 years ago. An expert in that subfield will tell you in 30 seconds.
   Without asking, you rediscover the same dead end.

2. **Writing forces clarity.** You think you understand your idea — until you try to write
   down the exact hypothesis, the exact experiment, and the exact success criteria. Vague
   intuitions collapse when you have to commit them to paper. This is a FEATURE.

3. **Feedback improves the experiment design, not just the idea.** Someone might say
   "your idea is good, but test it on X instead of Y" or "add ablation Z to make the
   result convincing." This makes your eventual paper stronger.

4. **It builds your network.** Every proposal you share is a conversation starter with
   someone who might become a collaborator, reviewer, or reference letter writer.

5. **It prevents sunk cost bias.** If you implement first, you're emotionally invested.
   You'll keep running experiments hoping it works instead of pivoting. A proposal has
   no sunk cost — you can drop it and move to the next idea painlessly.

**Who to share with (in priority order):**
- Your supervisor / advisor (always first)
- Lab members working on related topics
- Researchers you've met at conferences or online (cold email is fine — attach the 1-pager)
- Online communities: r/MachineLearning, Twitter/X ML community, lab Slack channels
- The original paper's authors (surprisingly responsive — they care about follow-up work)

```
I want to write a 1-2 page research proposal for a modification to {ALGORITHM}.

Context:
- I have a verified working implementation of {ALGORITHM} in {FILE}
- It achieves return > {X} on {ENV} at {N} steps (confirmed baseline)
- I have a deep understanding of the algorithm (see {EXPLAINED_MD_PATH})

My idea:
  {describe your idea in 3-5 sentences — what you want to change and why}

Informal intuition for why it might work:
  {your gut reasoning — what observation or analogy led you here}

Generate a research proposal with these sections:

1. TITLE (concise, specific — not "Improving Dreamer")

2. PROBLEM STATEMENT (1 paragraph)
   What specific limitation of {ALGORITHM} are you addressing?
   Give a concrete example where the current method fails or is suboptimal.

3. PROPOSED APPROACH (2-3 paragraphs)
   What exactly do you change? Be precise — which component, which equation,
   what replaces what. Include the modified equation(s) in KaTeX.
   Why should this work? What's the theoretical or intuitive justification?

4. RELATIONSHIP TO PRIOR WORK (1 paragraph)
   What's the closest existing work? How is your idea different?
   (List 3-5 most relevant papers — I'll verify these exist.)

5. EXPERIMENT PLAN
   - Environments to test on (easy + hard)
   - Baseline: vanilla {ALGORITHM} with exact hyperparameters
   - Ablations: what to turn on/off to isolate the effect
   - Metrics: what numbers prove it works (not just "higher return" — be specific)
   - Estimated compute: how many GPU-hours for the full experiment

6. EXPECTED OUTCOME
   Concrete prediction: "I expect X to improve from Y to Z because..."
   Also state: what result would DISPROVE the hypothesis?

7. RISK ASSESSMENT (1 paragraph)
   What could go wrong? What's the most likely failure mode?
   What's the fallback if the idea doesn't work as expected?

Format: clean markdown, under 2 pages when rendered. No filler.
Tone: confident but honest about uncertainties.

The goal is to share this with {3-5 experts} for feedback BEFORE implementing.
```

**Example filled in:**

```
I want to write a 1-2 page research proposal for a modification to Director.

Context:
- I have a verified working implementation of Director (hierarchy.py + agent.py)
- It achieves return > X on DMLab sparse-reward navigation
- I have a deep understanding (see DIRECTOR_EXPLAINED.md)

My idea:
  Replace the fixed skill duration K=8 with an adaptive mechanism.
  Add a "termination network" (like option-critic) that predicts when the
  Worker has reached the current goal, triggering an early Manager update.
  This lets the Manager set easy goals (terminated in 3 steps) and hard goals
  (held for 12 steps) depending on context.

Informal intuition:
  Fixed K=8 is wasteful — sometimes the Worker reaches the goal in 3 steps
  and wastes 5 steps doing nothing. Sometimes 8 steps isn't enough. Adaptive
  K should improve both efficiency and capability.
```

**Feedback collection template (email/message):**

```
Subject: Quick feedback on {TITLE}? (1-page proposal, 5 min read)

Hi {NAME},

I'm working on {ALGORITHM} and have an idea for {1-sentence summary}.
I've attached a short proposal (1 page) — would really appreciate your
quick reaction before I start implementing.

Specifically, I'm wondering:
1. Has something similar been tried? Am I missing related work?
2. Does the theoretical motivation hold, or is there a flaw I'm not seeing?
3. Is the experiment plan convincing?

No pressure for a detailed review — even a "this reminds me of paper X"
or "I don't think assumption Y holds" would be extremely helpful.

Thanks,
{YOUR NAME}
```

**LLM proposal review prompt (use BEFORE sending to humans):**

Run this as a sanity check. An LLM can't replace expert feedback (it doesn't
know what's been tried and abandoned, or what reviewers actually care about),
but it can catch structural weaknesses, missing baselines, and logical gaps
you might have missed while writing.

```
Review this research proposal as a critical but constructive ML reviewer.
You are an expert in {SUBFIELD, e.g. "model-based RL and hierarchical planning"}.

<proposal>
{paste your full proposal here}
</proposal>

Score each dimension 1-5 (1=fatal flaw, 3=acceptable, 5=excellent) and
explain your score in 1-2 sentences:

CHECKLIST:

□ NOVELTY (1-5)
  - Is this genuinely new, or a minor variation of existing work?
  - Has this exact idea (or something very close) been published before?
  - If it's incremental, is the increment meaningful?

□ CLARITY (1-5)
  - Can you state the hypothesis in one sentence after reading?
  - Is the proposed change precisely defined (which equation changes, how)?
  - Are there ambiguities that would cause two readers to imagine different experiments?

□ MOTIVATION (1-5)
  - Is the problem real? Does the baseline actually fail in the way described?
  - Is the failure example concrete and verifiable, or hand-wavy?
  - Would solving this problem matter to the community?

□ THEORETICAL SOUNDNESS (1-5)
  - Does the "why it should work" argument hold up?
  - Are there hidden assumptions? (e.g., "assumes reward is smooth" — is it?)
  - Could this modification break something else? (e.g., destabilise training)

□ EXPERIMENT DESIGN (1-5)
  - Are the environments well-chosen? (easy env for sanity + hard env for real test)
  - Is the baseline fair? (same hyperparams, same compute)
  - Are the ablations sufficient to isolate the effect?
  - Are the success metrics specific enough? ("return > X" not just "improves")
  - Is the compute estimate realistic?

□ PRIOR WORK (1-5)
  - Are the cited papers real and relevant? (flag any that seem hallucinated)
  - Is there obvious missing prior work? List any papers the author should read.
  - Is the differentiation from prior work convincing?

□ RISK & FALSIFIABILITY (1-5)
  - Is there a clear condition that would disprove the hypothesis?
  - Are the failure modes identified realistic?
  - Is the fallback plan reasonable?

OVERALL ASSESSMENT:
  - In 2 sentences: what is the strongest aspect of this proposal?
  - In 2 sentences: what is the single biggest weakness?
  - Verdict: [IMPLEMENT AS-IS / REVISE AND RESUBMIT / RETHINK IDEA]
  - If REVISE: list the 3 most important changes to make before implementing.

MISSING RELATED WORK:
  List 3-5 papers the author should read before proceeding.
  Format: "Title (Author, Year) — relevant because {reason}"
  IMPORTANT: Only list papers you are confident actually exist.
  If unsure whether a paper exists, say so explicitly.
```

**How to use the LLM review:**
1. Run the review prompt → read the scores and comments
2. Fix anything scored 1-2 (fatal or weak) before sharing with humans
3. Treat the "Missing Related Work" section with skepticism — verify every
   paper the LLM suggests actually exists (LLMs hallucinate citations)
4. The LLM review is a **spell-check for logic**, not a replacement for
   expert feedback. Low LLM scores = definitely fix. High LLM scores ≠
   definitely good (the LLM can't know what's been tried and failed).

---

### Prompt 10: Research Modification on Verified Baseline

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

## Content Publishing Pipeline

The workflow doesn't end at working code. To maximise impact, each algorithm
gets a **content stack** — from deep technical blog to short-form social media.

### The Content Stack

```
┌──────────────────────────────────────────────────────┐
│  TIER 1: *_EXPLAINED.md (Technical Blog Post)         │
│  Deep, authoritative, 3000-5000 words                │
│  Audience: ML researchers + serious practitioners     │
│  Publish: personal blog, GitHub, Distill-style site   │
├──────────────────────────────────────────────────────┤
│  TIER 2: Animated Explainer Video (3Blue1Brown style) │
│  10-20 minutes, visual math animations               │
│  Audience: technical YouTube audience                 │
│  Publish: YouTube                                     │
├──────────────────────────────────────────────────────┤
│  TIER 3: Short-Form Hook (TikTok / Reels / Shorts)   │
│  30-90 seconds, one surprising insight + meme energy  │
│  Audience: broader tech/science audience              │
│  Publish: TikTok, YouTube Shorts, Instagram Reels     │
└──────────────────────────────────────────────────────┘
```

**Key principle:** Content flows DOWN the stack. The EXPLAINED.md is the
source of truth. The video script is distilled from it. The short-form hook
is one scene extracted from the video.

### Tier 1: Blog Post Publishing

#### Tools
- **Static site:** Hugo, Jekyll, or Next.js + MDX (for KaTeX rendering)
- **Hosting:** GitHub Pages (free), Vercel, or Cloudflare Pages
- **Domain:** `yourdomain.dev` or similar

#### From EXPLAINED.md to Published Blog

```
1. Copy *_EXPLAINED.md content
2. Add YAML frontmatter (title, date, tags, description)
3. Convert relative code links to GitHub permalink URLs
4. Add hero image (architecture diagram from paper or your own)
5. Add Open Graph meta tags for social sharing preview
6. Deploy via git push
```

#### Frontmatter template:

```yaml
---
title: "DreamerV1: Every Design Choice Explained from Scratch"
date: 2026-04-15
author: "Your Name"
tags: [reinforcement-learning, world-models, dreamer, model-based-rl]
description: "A Feynman-style deep dive into every trick in DreamerV1, from RSSM to lambda returns, with code references."
math: true
toc: true
---
```

#### Cross-posting (for reach):
- Main blog (canonical URL)
- Medium (with canonical link back)
- dev.to (with canonical link back)
- Reddit r/MachineLearning (as a text post with link)
- Twitter/X thread (10-tweet summary of key insights)

---

## Full Content Lifecycle per Algorithm

```
PHASE 1 — UNDERSTAND (Prompt 1)
  Explain every design choice → personal comprehension check

PHASE 2 — CODE (Prompts 2-5)
  Write code → shape test → smoke test → full train → verify

PHASE 3 — TEACH (Prompt 6)
  Write *_EXPLAINED.md → the source of truth for all content

PHASE 4 — PUBLISH BLOG (Tier 1)
  Add frontmatter → deploy to blog → cross-post to Medium/Reddit/X

PHASE 5 — LONG VIDEO (Prompt 7)
  Script + Manim scenes → render → voiceover → YouTube

PHASE 6 — SHORT CONTENT (Prompt 8)
  3 TikTok scripts → pick best hook → film/animate → post

PHASE 7 — PROPOSE (Prompt 9)               ← BEFORE implementing your idea
  Write 1-page proposal → share with 3-5 experts → collect feedback
  Gate: only proceed to Phase 8 after incorporating feedback

PHASE 8 — RESEARCH (Prompt 10)
  Implement modification → A/B test → analyse → write up results

Timeline per algorithm:
  Phase 1-2: Day 1-3 (understand + code)
  Phase 3:   Day 3   (write the EXPLAINED.md)
  Phase 4:   Day 4   (publish blog post)
  Phase 5:   Day 5-7 (video production — can be parallel)
  Phase 6:   Day 4   (short content — fast, do alongside blog)
  Phase 7:   Day 7-9 (proposal + feedback loop — DO NOT SKIP)
  Phase 8:   Day 10+  (implement, only after proposal cleared)
```

---

## Common Pitfalls Checklist

Check these before starting a full training run:

### Off-by-one Errors (Dreamer-style)
- [ ] Reward at time $t$ corresponds to the transition $s_t 	o s_{t+1}$, not $s_{t-1} 	o s_t$
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
Day 1:   Read paper → Prompt 1 (explain) → understand every component
          ↓
Day 1:   Prompt 2 or 3-Layer1 (skeleton) → shape tests pass
          ↓
Day 2:   Layer 2 (training) → smoke test passes
          ↓
Day 2:   Layer 3 (full loop) → quick train matches paper on easy env
          ↓
Day 3:   Full training run → acceptance criteria met
          ↓
Day 3:   Prompt 6 → write *_EXPLAINED.md (the source of truth)
          ↓
Day 4:   Publish blog post (Tier 1) + TikTok hooks (Prompt 8)
          ↓
Day 5-7: 3B1B video script (Prompt 7) → Manim → render → YouTube
          ├──────────────────────────────────────────────────────────
          │  YOUR OWN RESEARCH STARTS HERE
          ↓
Day 7:   Prompt 9 → write 1-page proposal for your idea
          ↓
Day 8-9: Share proposal with supervisor, labmates, experts → collect feedback
          ↓
Day 10:  Revise idea based on feedback (or pivot to a different idea)
          ↓
Day 10+: Prompt 10 (your modification) → A/B test → research result
          ↓
          Repeat for next paper in the curriculum
```

For simple algorithms (DQN, PPO): Layers 1-3 collapse into a single Prompt 2 call.
For complex algorithms (Dreamer, Director): Always do layers separately.
For content: EXPLAINED.md is always written. Video + TikTok are optional but high-leverage.
**For your own ideas: ALWAYS write a proposal (Prompt 9) before implementing (Prompt 10).**

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
├── research-world-models/
│   ├── phase4_dreamer/
│   │   ├── 04_dreamer_claude.py   ← layered build: WM → AC → loop
│   │   └── DREAMER_EXPLAINED.md   ← Prompt 6 output → Tier 1 blog source
│   └── phase5_director/
│       ├── 05_director_claude.py  ← layer on top of verified Dreamer
│       └── DIRECTOR_EXPLAINED.md
├── blog/                          ← Hugo/Next.js site source (Tier 1)
│   └── content/posts/
│       ├── dreamer-explained.md   ← EXPLAINED.md + frontmatter
│       └── director-explained.md
└── videos/                        ← Manim + short-form assets (Tier 2-3)
    ├── dreamer/
    │   ├── script.md              ← Prompt 7 output
    │   ├── scenes/                ← Manim .py files, one per scene
    │   └── shorts/                ← TikTok scripts (Prompt 8 output)
    └── director/
```

---

## Quick Reference: Which Prompt to Use

| Situation | Prompt |
|-----------|--------|
| Understanding a paper before coding | **Prompt 1** |
| New simple algorithm (DQN, PPO, C51, SAC) | **Prompt 2** |
| New complex algorithm — world model piece | **Prompt 3, Layer 1** |
| New complex algorithm — actor-critic piece | **Prompt 3, Layer 2** |
| New complex algorithm — full training loop | **Prompt 3, Layer 3** |
| Adding hierarchy to existing algorithm | **Prompt 4** |
| Something is broken during training | **Prompt 5** |
| Distill-style EXPLAINED.md blog post | **Prompt 6** |
| 3Blue1Brown video script + Manim spec | **Prompt 7** |
| TikTok / Reels short-form hooks | **Prompt 8** |
| Research proposal before implementing | **Prompt 9** |
| Testing my own research idea | **Prompt 10** |
