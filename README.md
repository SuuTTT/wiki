# Wiki - RL Learning & Research Repository

Welcome to the Reinforcement Learning research repository. 

* **Purpose:** This directory contains foundational tutorials learning from seminal research repositories (like CleanRL) to bridge the gap toward leading autonomous novel research.
* **Methodology:** We build a series of well-commented, CleanRL-style `*tutorial.py` files featuring incremental mathematical implementations, minimum dependency and clean Python code. We will continue this methodology until we are fully equipped to code our own new research architectures from scratch.(finally create a reading list of code in difficult order, building everything from scratch in first principle,for a beginner to learn gradually)
* **Execution & Logging Standards:** 
  - Each tutorial is designed to run independently (e.g., `python learn-cleanrl/ppo_tutorial.py`).
  - To properly measure success and performance bounds, all metrics are automatically intercepted by the custom `RLTracker` utility (found in `learn-cleanrl/cleanrl_utils/logger.py`). 
  - The results are cleanly logged in standardized directories under `benchmark/<algorithm>/<YYYY-MM-DD-timestamp>` for native TensorBoard metric visualization and automatic `.pth` model checkpointing.

## Current Research Focus

The current trajectory of this workspace is directed specifically toward solving sparse and temporally extended environments. Our core active research branches are:

1. **Representation Learning for RL** (Learning dense, abstracted, data-efficient feature spaces directly from unstructured inputs).
2. **Learning Abstraction & World Models for Hierarchical RL and Planning** (Temporal abstraction, The Options Framework, Unsupervised latent skill discovery like DIAYN, and generative predictive simulators like Dreamer).

For the ongoing algorithm roadmap, please see `learn-cleanrl/docs/TODO.md`. For a breakdown of the math/theory, refer to the documentation gathered inside `learn-cleanrl/docs/`.
