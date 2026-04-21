# Section 1: Introduction

Modern engineering systems increasingly demand control paradigms that can handle high-dimensional observations while maintaining theoretical robustness. While **Temporal Difference Model Predictive Control (TD-MPC2)** has demonstrated state-of-the-art performance in complex continuous control tasks, it remains fundamentally limited by its reliance on a "flat" latent world model. In long-horizon tasks, the compounding errors of primitive-action rollouts and the absence of structural awareness lead to highly inefficient exploration and planning.

To address these challenges, we propose a unified modeling-control framework that integrates **Structural Information Theory (SIT)** with latent-space trajectory optimization. SIT provides a principled mathematical foundation for partitioning a complex system's state space into a hierarchical **Encoding Tree** based on the 2-dimensional structural entropy of the observed transition dynamics. 

Our core contribution, **SIT-MPC**, leverages this discovered topology in two ways:
1.  **Topological Anchor Points**: SIT naturally identifies bottleneck states and module centroids, which serve as stable subgoals for the model's high-level planning.
2.  **Jumpy-MPC Planning**: By planning over macro-states defined by the Encoding Tree, our approach reduces the effective search horizon, allowing for more robust long-range navigation.

We evaluate our approach on a suite of challenging environments, including sparse-reward 8x8 GridWorlds and continuous-action CarRacing. Our results demonstrate that SIT-MPC significantly outperforms flat model-based baselines in exploration efficiency and goal-reaching stability, offering a promising path toward verified data-driven control of complex autonomous systems.
