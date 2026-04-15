# Deep Hierarchical Planning & Abstract World Models
*Based on Danijar Hafner's Dreamer and Director Architectures*

## 1. How Danijar Defines Hierarchical Planning
In the context of **Director (2022)**, Danijar Hafner defines hierarchical planning as the ability to break down long-horizon, sparse-reward tasks into manageable subgoals directly from high-dimensional inputs (pixels), without relying on manually engineered state spaces.

Director archives this through a two-tiered hierarchy operating inside the latent space:
*   **The Manager (High-Level):** Operates at a slower temporal scale (e.g., updating every $K=8$ steps). It observes the latent state and outputs a *latent goal*. It is optimized to maximize the sparse extrinsic reward of the environment plus an intrinsic exploration bonus.
*   **The Worker (Low-Level):** Operates at every single time step. It is conditioned on the goal provided by the Manager and outputs actual environment actions. It is optimized to maximize an *intrinsic reward*—specifically, the cosine similarity between its current state feature and the Manager's requested goal feature.
*   **The Goal Autoencoder:** To ensure the Manager doesn't request unrealistic or impossible latent states, Director enforces a goal autoencoder. The Manager outputs a continuous vector, which is discretized and decoded into a valid latent feature.

Hierarchical planning here is defined as **Manager proposing latent subgoals $\rightarrow$ Worker executing motor commands to reach them**, entirely within the "imagination" of the learned world model.

## 2. Relationship to Dreamer
Yes, Director is heavily and directly based on **Dreamer**. 
*   **World Model Base:** Director uses the exact Recurrent State Space Model (RSSM) introduced in Dreamer to compress pixels into a low-dimensional latent space.
*   **Imagination:** Just like Dreamer, both the Manager and Worker policies are trained purely in latent imagination using straight-through gradients or REINFORCE. 
*   Director essentially takes Dreamer, copies the Actor-Critic module, creates two of them (Manager and Worker), and connects them via a structural bottleneck (the latent goal).

---

## 3. Brainstorming: Future Research Directions
Based on the foundations of Dreamer and Director, we can explore several novel directions for our own research in Hierarchical Reinforcement Learning (HRL) and Abstract World Models.

### Idea A: Jumpy Abstract World Models (Temporal State Abstraction)
*   **Problem in Director:** While the Manager *issues* goals every $K$ steps, the World Model (RSSM) still unrolls every single intermediate 1-step transition to evaluate the Worker. This is computationally expensive.
*   **Our Idea:** Train a **Jumpy World Model** that skips steps. The Manager's world model should only predict the transition $S_t \rightarrow S_{t+K}$ directly.
*   **Research Question:** Can we build a hierarchical RSSM where the higher level predicts "abstract states" at a coarse timescale, bypassing the need to simulate low-level physics during high-level planning?

### Idea B: Adaptive Temporal Abstraction (Dynamic $K$)
*   **Problem in Director:** Director uses a fixed step size ($K=8$ or $K=16$) for the Manager's goal update.
*   **Our Idea:** The agent should learn *when* to set a new goal. Using intrinsic metrics like **Latent Entropy**, **Prediction Error**, or **Information Bottlenecks**, the Manager can issue a new goal only when a "sub-task" is completed (e.g., reaching a door) rather than arbitrarily every 8 steps.
*   **Research Question:** How can we train a differentiable gating mechanism that allows the Manager to dictate its own time horizons dynamically?

### Idea C: Semantic / Language-Conditioned Goal Autoencoders
*   **Problem in Director:** The Goal Autoencoder just compresses states. The goals are vectors that can be visualized but have no semantic grounding.
*   **Our Idea:** Tie the Goal Autoencoder to a pre-trained VLM (Vision-Language Model) like CLIP or a lightweight captioner. The Manager would output abstract vectors that correspond heavily to language concepts (e.g., latent vector for "get wood").
*   **Research Question:** Can grounding the latent goal space with semantic language prior heavily accelerate exploration in sparse-reward environments?

### Idea D: Unsupervised Skill Discovery (Options Framework in Latent Space)
*   **Problem in Director:** The Worker is just a single policy trying to hit a continuous/discrete vector.
*   **Our Idea:** Discretize the Worker into $N$ distinct "Skills" (similar to the Options framework or DIAYN - Diversity is All You Need). The Manager outputs a discrete categorical integer corresponding to a Skill, and the chosen Skill policy executes until termination.
*   **Research Question:** How can we cluster the latent space dynamics of an RSSM into repeatable, reusable skills without extrinsic rewards?

### Idea E: Multi-Scale Hierarchies (3+ Levels)
*   **Problem in Director:** It only uses 2 levels. For deeply complex tasks (like Minecraft or vast 3D navigation), 2 levels might not bridge the gap between 1 million pixel steps and the grand objective.
*   **Our Idea:** A hierarchical RSSM where Level 3 operates at $K=64$, Level 2 at $K=8$, and Level 1 at $K=1$.
*   **Research Question:** Can we stack Goal Autoencoders to create nested abstractions, and how do we stabilize the cascading actor-critic losses?

---

## 4. Next Steps for our "CleanRL" Implementation
To pave the way for these research ideas, our immediate Next Phase is to build a hyper-readable, easily modifiable Director baseline:
1. **Merge DreamerV3 + Director:** We will use the advanced components of DreamerV3 (Symlog, LayerNorm, Discrete RSSM) but architect it into a Director-style Manager-Worker split.
2. **Implement Goal Autoencoder:** Add the discrete projection network over the RSSM representation.
3. **Dual Actor-Critic Loop:** Implement two PPO/REINFORCE instances where the Worker's reward is strictly the latent cosine similarity to the Manager's output.
