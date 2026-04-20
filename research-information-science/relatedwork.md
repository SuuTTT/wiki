The **TD-MPC (Temporal Difference Learning for Model Predictive Control)** line of work represents a prominent family of model-based reinforcement learning (MBRL) algorithms specifically designed for highly scalable and robust continuous control. Unlike the Dreamer family, which relies heavily on actor-critic methods and often reconstructs visual observations, the TD-MPC family utilizes an "implicit" latent world model combined with online trajectory optimization.

Here is a detailed breakdown of the TD-MPC lineage, including its foundational architecture and its recent temporal and hierarchical extensions:

**1. The Foundation: TD-MPC and TD-MPC2**
*   **Core Architecture:** TD-MPC2 captures environment dynamics entirely within a latent space without using a decoder to reconstruct raw, high-dimensional observations (like pixels). This saves massive computational resources since visually rendering the future is often unnecessary for control.
*   **Components:** The architecture consists of five neural network components (mostly Multilayer Perceptrons): an observation encoder, a dynamics model, a reward model, a terminal-value model, and a policy. 
*   **Training & Planning:** These components are jointly optimized using a combination of self-supervised consistency loss, supervised reward loss, and supervised temporal-difference (TD) loss for the value function. Once the latent dynamics are learned, TD-MPC2 uses a Model Predictive Control (MPC) planner—specifically Model Predictive Path Integral (MPPI)—to roll out latent predictions and perform local trajectory optimization to generate actions at inference time.

**2. The Temporal Extension: TAWM (Time-Aware World Model)**
*   **The Problem:** Standard TD-MPC2 assumes a fixed time-step size (e.g., operating strictly at 50Hz) during training. This leads to temporal resolution overfitting, compounding errors, and poor generalization if the model is deployed in a real-world setting that operates at a different or variable frequency. 
*   **The TAWM Solution:** TAWM builds directly on the TD-MPC2 architecture but explicitly conditions the dynamics and reward models on the time-step size, $\Delta t$. Instead of learning a direct state-to-state transition, TAWM learns the latent-state derivative (gradient) and advances the state using numerical integration methods, such as Euler or fourth-order Runge-Kutta (RK4) integration.
*   **Training:** By sampling observations at a mixture of different $\Delta t$ intervals (e.g., using a log-uniform distribution), TAWM learns multi-scale behaviors—from fast to slow dynamics—simultaneously. This allows the agent to make flexible, single-step predictions across various temporal gaps at inference time without requiring any additional training data.

**3. The Hierarchical Extension: SPlaTES (Stable Planning with Temporally Extended Skills)**
*   **The Problem:** Because TD-MPC2 plans using primitive (single-step) actions, it still struggles with long-horizon tasks due to the compounding error problem, where small inaccuracies in the world model's predictions snowball over extended rollouts. 
*   **The SPlaTES Solution:** SPlaTES upgrades the continuous control paradigm by doing MPC over *temporally extended skills* rather than primitive actions. The agent uses an abstract skill world model to predict the outcome of a skill in an abstract state space. 
*   **Planning & Execution:** SPlaTES uses an MPC planner (like iCEM) to find the optimal sequence of skills. The lower level then executes the chosen skill, which inherently compensates for physical perturbations and drifts that a primitive-action planner like standard TD-MPC2 would fail to recover from. Interestingly, the robust behaviors learned by SPlaTES can even be distilled back into a standard TD-MPC2 agent.



# links
Based on the provided sources, here are the details and available links for the papers and projects in the TD-MPC line of work:

**1. TAWM (Time-Aware World Model)**
*   **Paper Title:** Time-Aware World Model for Adaptive Prediction and Control
*   **Code/Project Link:** [github.com/anh-nn01/Time-Aware-World-Model](https://github.com/anh-nn01/Time-Aware-World-Model)

**2. TD-MPC and TD-MPC2**
*   **Paper Titles:** 
    *   *TD-MPC:* Temporal Difference Learning for Model Predictive Control (ICML 2022). https://github.com/nicklashansen/tdmpc
    *   *TD-MPC2:* TD-MPC2: Scalable, Robust World Models for Continuous Control (ICLR 2024).
*   **Links:** https://github.com/nicklashansen/tdmpc2

**3. SPlaTES (Stable Planning with Temporally Extended Skills)**
*   **Paper Title:** Long-Horizon Planning with Predictable Skills https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_136.pdf
*   **Links:** https://nicoguertler.github.io/splates-pages/

