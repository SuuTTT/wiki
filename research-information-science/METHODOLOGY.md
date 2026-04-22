# Section 3: Methodology - Structural Information-Theoretic State Abstraction

The core of our approach lies in the integration of **Structural Information Theory (SIT)** with the latent transition model of **TD-MPC2**. While TD-MPC2 excels at planning with a flat latent representation, it lacks the topological awareness required for long-horizon stability. We introduce a **Heat-Kernel Structural Entropy Loss** to regularize the latent space.

### 3.1. Baseline: TD-MPC2
The standard TD-MPC2 learns a world model consisting of:
- **Encoder**: $z_t = \phi(o_t)$
- **Dynamics**: $z_{t+1} = d(z_t, a_t)$
- **Inference**: MPPI-based planning in $z$-space.

The loss function is purely predictive:
$$\mathcal{L}_{TD} = \lambda_1 \mathcal{L}_{consistency} + \lambda_2 \mathcal{L}_{reward} + \lambda_3 \mathcal{L}_{value}$$

### 3.2. SIT Integration
We augment the TD-MPC2 objective with a structural loss $\mathcal{L}_{SIT}$ that operates on the latent transition graph $G$ constructed from the prediction horizon $H$.

#### 3.2.1. Adjacency via Heat Kernel
For a rollout of latent states $\{z_0, \dots, z_H\}$, we compute a temporal adjacency matrix $A$ using a Gaussian kernel:
$$A_{ij} = \exp\left(-\frac{\|z_i - z_j\|^2}{2\sigma^2}\right)$$
This matrix $A$ represents the "topological proximity" of states within the world model's imagination.

#### 3.2.2. Structural Entropy Calculation (H1 & H2)
To align with the formal SIT definition provided in the [Structural Information Theory](Structural_information.md) documentation, we differentiate between 1D and 2D entropy:

**1-Dimensional Structural Information ($H^{(1)}$):**
Measures the baseline uncertainty of the latent transition graph without hierarchical abstraction.
$$H_1(G) = -\sum_{i=1}^n \frac{d_i}{V} \log_2 \frac{d_i}{V}$$
where $d_i$ is the node degree (from heat kernel) and $V$ is the total volume.

**2-Dimensional Structural Information ($H^{(2)}$):**
We approximate the **Encoding Tree** $T$ by partitioning the transition graph into temporal modules (Macro-States). The 2D entropy $H_T(G)$ quantifies the uncertainty remaining after this abstraction:
$$H^{(2)}(G, T) = \sum_{\alpha \in T, \alpha \neq \lambda} - \frac{g_\alpha}{V} \log_2 \frac{V_\alpha}{V_{\alpha^-}}$$
where:
- $g_\alpha$: Cut weight (edges crossing boundary of module $\alpha$).
- $V_\alpha$: Volume of module $\alpha$.
- $V_{\alpha^-}$: Volume of parent module (for our 2-level tree, this is $V$).

In our implementation, we use a **Temporal Bottleneck Partition** (splitting the prediction horizon $H$ into $\{[0, H/2), [H/2, H]\}$ modules) to verify if the world model naturally clusters states into stable temporal segments.

## 3.4. Reproducible Algorithm Specification: SIT-TD-MPC2

To ensure reproducibility, we detail the internal mechanics of the **SIT Transition Graph** construction and the **H2 Structural Entropy** derivation.

### Step 1: Latent State Rollout Generation
The algorithm begins by sampling a batch of $B$ sequences from the replay buffer. For each sequence, the world model performs a latent rollout of horizon $H$ (typically $H=5$):
1.  **Encode**: $z_0 = \phi(o_t)$
2.  **Predict**: $\hat{z}_{1 \dots H}$ is generated via the dynamics model $d(z_k, a_k)$.
3.  **Collection**: This results in a tensor $\mathbf{Z} \in \mathbb{R}^{B \times (H+1) \times L}$, representing the "imagined trajectory" of the agent.

### Step 2: Build the Transition Graph $G$
For each batch element $b \in B$, we construct a fully connected weighted adjacency matrix $A^{(b)}$:
1.  **Pairwise Distance**: Compute $D_{ij} = \|z_i - z_j\|^2$ for all pairs $i, j \in \{0 \dots H\}$.
2.  **Heat Kernel Weighting**: Apply the Gaussian kernel to convert distances to topological proximity:
    $$A_{ij} = \exp\left(-\frac{D_{ij}}{2\sigma^2}\right)$$
3.  **Self-Loop Removal**: Set $A_{ii} = 0$ to focus on state-to-state transitions.
4.  **Graph Statistics**:
    -   **Node Degree**: $d_i = \sum_j A_{ij}$
    -   **Graph Volume**: $V = \sum_i d_i$

### Step 3: Define the Temporal Encoding Tree $T$
We impose a hierarchical abstraction by partitioning the nodes into a 2-level encoding tree:
-   **Root ($\lambda$)**: The entire rollout $V$.
-   **Modules ($\alpha$)**: We split the horizon into two temporal partitions:
    -   $\alpha_{early} = \{z_0, \dots, z_{\lfloor H/2 \rfloor}\}$
    -   $\alpha_{late} = \{z_{\lfloor H/2 \rfloor + 1}, \dots, z_H\}$
-   **Leaves**: Individual latent states $z_k$.

### Step 4: Compute Structural Entropy Losses
1.  **1D Positioning Entropy ($H^{(1)}$)**:
    $$H^{(1)}(G) = -\sum_{i=0}^H \frac{d_i}{V} \log_2 \left(\frac{d_i}{V}\right)$$
    *Objective: Distribute latent states to avoid "collapsed" representations.*

2.  **2D Structural Entropy ($H^{(2)}$)**:
    Based on the tree $T$, we calculate the remaining uncertainty:
    $$H^{(2)}(G, T) = \sum_{\alpha \in \{\alpha_{early}, \alpha_{late}\}} \left[ - \frac{g_\alpha}{V} \log_2 \left(\frac{V_\alpha}{V}\right) \right]$$
    where:
    -   $g_\alpha$ is the sum of weights leaving module $\alpha$ (the "cut").
    -   $V_\alpha$ is the sum of degrees within module $\alpha$.
    *Objective: Maximize intra-module connectivity while minimizing inter-module "leakage."*

### Step 5: Joint Optimization
The final loss gradient is backpropagated through both the **Dynamics Model** and the **Encoder**:
$$\theta \leftarrow \theta - \eta \nabla_\theta \left( \mathcal{L}_{TD-MPC} + \beta (H^{(1)} + H^{(2)}) \right)$$
where $\beta$ is the `sit_coef`. This forces the world model to "imagine" trajectories that are not only reward-accurate but topographically structured.


The total loss becomes:
$$\mathcal{L}_{Total} = \mathcal{L}_{TD} + \beta \mathcal{L}_{SIT}$$
where $\beta$ is the `sit_coef`. This forces the dynamics model $d(z, a)$ to not only predict the next state but to ensure the resulting latent trajectory maintains high structural integrity.
