# Structural Information-Theoretic Director (SIT-Director)
*A Unified Framework for Abstract World Models, Hierarchical Planning, and Skill Discovery*

## 1. Introduction: The Synergy of SIT and Subgoal Planning
Director (Hafner et al., 2022) demonstrated that deep hierarchical planning from pixels is possible by splitting policies into a slow-timescale Manager and a fast-timescale Worker operating within a latent goal space. However, Director relies on arbitrary heuristics: a fixed goal-update interval ($K=8$), a rigid two-level hierarchy, and an unstructured goal autoencoder. 

**Structural Information Theory (SIT)** provides the exact mathematical language required to resolve these heuristics. By modeling the environment transition dynamics as a graph $G$ and the state abstraction as an **Encoding Tree $T$**, we can optimize hierarchical planning entirely via **Structural Entropy Minimization**. 

Here, we propose a novel architecture—**SIT-Director**—that maps the brainstormed concepts (Jumpy World Models, Adaptive Abstraction, Skill Discovery) directly into SIT principles.

---

## 2. Core Architecture: The Encoding Tree as the Goal Autoencoder
In standard Director, the Goal Autoencoder maps continuous latent states into discrete codes. In SIT-Director, we replace this opaque autoencoder with a dynamic **Encoding Tree $T$** built directly on top of the RSSM's latent space.

1. **The Latent Graph $G = (V, E, W)$:**
   As the agent explores, the RSSM generates sequential latent states $z_t$. We treat a running buffer of these latent states as the vertices $V$. The transition probabilities computed by the World Model's dynamics predictor $P(z_{t+1} | z_t, a_t)$ dictate the directed edge weights $W$.
2. **The Abstraction Hierarchy $T$:**
   The tree $T$ clusters the base latent states $z_t$ (leaves) into nested macro-states $\alpha$ (internal nodes). 
   * **Manager's Goal Space:** The nodes at depth $K$ of the encoded tree act naturally as the Manager's discrete subgoals. 
   * **Optimization:** The tree is not static. During the World Model training phase, we iteratively apply SIT operators (**merging** strongly connected latent states, **splitting** unstable ones) to minimize the $K$-dimensional structural entropy $H_T(G)$. This maximizes the **Decoding Information $D_T(G)$**, ensuring the subgoals correspond to the true "natural joints" of the environment.

---

## 3. Realizing the Research Goals via SIT

### A. Adaptive Temporal Abstraction (Solving the Fixed $K$ Problem) (Similar to SISL segment demo video, here we segment timescale) (we do not need demonstration expert data, learning abstract from dream)
* **The Problem:** Standard Director forces the Manager to pick a new goal exactly every 8 steps, cutting off skills prematurely or reacting too slowly.
* **The SIT Solution:** In SIT, a macro-state $\alpha$ is a distinct module of the graph containing a volume of states $V_\alpha$ and a boundary "cut" $g_\alpha$ (transitions exiting the module).
* **Mechanism:** The Manager selects a macro-state $\alpha$ as the goal. The Worker executes actions to traverse the latent space within $\alpha$. The Manager *only* issues a new goal when the RSSM's latent state $z_t$ crosses the boundary cut $g_\alpha$ into a new module. The temporal chunking becomes entirely dynamic, driven by the geometry of the environment rather than a clock.

### B. Jumpy Abstract World Models (Temporal State Abstraction) (Similar to SISA)
* **The Problem:** RSSMs unroll every micro-step to predict the future, wasting compute on trivial low-level physics.
* **The SIT Solution:** Once the Encoding Tree $T$ is formed, the environment can be modeled at the macro-state level. 
* **Mechanism:** We train a **High-Level Transition Model** that predicts $P(\alpha_{t+1} | \alpha_t, \text{ManagerAction})$. Because SIT guarantees that transitions within $\alpha$ have low cross-module entropy, the Manager can reliably plan jumps from room to room ($\alpha_1 \to \alpha_2 \to \alpha_3$) without computing the millimeter-by-millimeter footsteps (leaves) inside them.

### C. Unsupervised Skill Discovery (Options Framework in Latent Space) *(this is new, since it combines abstract state and action)
* **The Problem:** Discovering distinct, repeatable skills without extrinsic rewards.
* **The SIT Solution:** A "skill" is mathematically equivalent to a trajectory that reliably navigates to and terminates at a specific module boundary in $T$.
* **Mechanism:** By maximizing $D_T(G) = H_1(G) - H_T(G)$, the tree inherently isolates bottlenecks (e.g., doorways, object pickups) because they minimize the transition uncertainty globally. The Manager simply selects $\alpha$ (the module), and the Worker's Policy is conditioned on reaching the boundary of $\alpha$. These modules $\alpha$ intrinsically represent the discovered skills.

### D. Multi-Scale Hierarchies ($N$-Level Director) *(this is new since our original work also only has 2 levels)  (later)
* **The Problem:** Two levels of hierarchy fail on massive tasks (e.g., Minecraft).
* **The SIT Solution:** The Encoding Tree $T$ is natively recursive. 
* **Mechanism:** Instead of just a Manager and Worker, we deploy agents corresponding to levels of the tree. 
  * Level 1 (Leaves): Motor control $z_t \to a_t$
  * Level 2 (Module $\alpha$): Room navigation $\alpha \to \text{target } \alpha$
  * Level 3 (Module $\beta$, parent of $\alpha$): Biome traversal $\beta \to \text{target } \beta$
  Each level passes its selected node down as a spatial target for the level below.

---

## 4. Proposed Training Loop for SIT-Director

To bridge CleanRL and Structural Information Theory:

1. **Step 1: Collect Experience & Train RSSM (Standard)**
   The agent collects states, actions, and rewards. The standard DreamerV3 continuous/discrete RSSM predicts $z_t$.
2. **Step 2: Optimize the Encoding Tree $T$ (SIT Optimization)**
   Over the batch of latent trajectories, calculate the adjacency weights $W_{i,j}$ based on the RSSM transition likelihoods. Apply local SIT operators (Merge, Split, Lift) to update the Tree $T$, maximizing $D_T(G)$. **(this is new since original SISA paper create edge according to frequency of transition, we can create edge according to the transition probability predicted by RSSM, which is more accurate and stable)**
3. **Step 3: Worker Optimization (Intrinsic Alignment)**
   The Worker receives the target module $\alpha^*$ from the Manager. Its intrinsic reward is proportional to the log-probability of transitioning closer to the boundary $g_{\alpha^*}$. It is trained via REINFORCE or PPO.
4. **Step 4: Manager Optimization (Extrinsic Maximization)**
   The Manager, running at the abstract scale of tree modules, receives the sparse extrinsic rewards from the environment. It uses an Actor-Critic loop to select the optimal sequence of modules $\alpha_1 \to \alpha_2$.

## 5. Conclusion
By fusing Hafner's latent-imagination architecture with Structural Information Theory, we eliminate the need for temporal heuristics and rigid network constraints. SIT provides an exact, globally optimal metric $H_K(G)$ for building hierarchical abstractions, allowing AI and robotic agents to discover true hierarchical plans in chaotic, pixel-based environments.