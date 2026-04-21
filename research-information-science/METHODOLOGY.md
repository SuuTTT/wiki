# Section 3: Methodology - Structural Information-Theoretic State Abstraction

The core of our approach lies in the integration of **Structural Information Theory (SIT)** with the latent transition model of **TD-MPC2**. Unlike standard encoders that produce a flat representation, our model constructs a hierarchical **Encoding Tree $\mathcal{T}$** based on the 2-dimensional structural entropy of the latent transition graph.

### 3.1. Latent Transition Graph Construction
Let $z_t \in \mathcal{Z}$ be the latent state at time $t$. We maintain a transition graph $G = (V, E)$, where $V$ represents a set of quantized latent states and $E$ represents the observed transitions $z_t \xrightarrow{a_t} z_{t+1}$. The edge weights $w_{ij}$ are proportional to the transition frequency between latent nodes $i$ and $j$.

### 3.2. Structural Entropy Minimization
We seek to find an Encoding Tree $\mathcal{T}$ that minimizes the 2-dimensional structural entropy $H^{(2)}(G, \mathcal{T})$:

$$H^{(2)}(G, \mathcal{T}) = - \sum_{\alpha \in \mathcal{T}, \alpha \neq \text{root}} \frac{g_\alpha}{vol(G)} \log_2 \frac{vol(\alpha)}{vol(parent(\alpha))}$$

where:
- $g_\alpha$ is the cut weight (sum of edge weights crossing the boundary of module $\alpha$).
- $vol(\alpha)$ is the sum of degrees of all nodes within the subtree $\alpha$.
- $vol(G)$ is the total volume of the entire graph.

**By minimizing this entropy, we naturally discover the "topological joints" of the environment.**

### 3.3. Jumpy-MPC Planning
The discovered tree $\mathcal{T}$ provides a multi-resolution space for planning. Instead of a fixed primitive-action horizon $H$, our SIT-MPC planner operates over **Macro-States** (internal nodes of $\mathcal{T}$). When the agent is within a high-connectivity module (low local entropy), the MPC can "jump" directly to a boundary state, effectively solving the search problem in the abstract space before refining primitive actions in the local vicinity.
