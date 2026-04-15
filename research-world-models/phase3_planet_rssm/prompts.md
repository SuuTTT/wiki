# whyt the SPS being 17 currently, is it normal? 
how long will it take to train then? how many step and time does original paper train

**Is an SPS (Steps Per Second) of 17 normal?**

Yes, it is entirely normal for this specific implementation, and it highlights the exact theoretical bottleneck of the **PlaNet (Phase 3)** architecture. 

In our previous Phase 1 (PPO + VAE), the agent just passed the image through a CNN, got an action, and stepped the environment. It was computationally lightweight at inference time, leading to high SPS.

In PlaNet, however, there is **no policy network**. We are doing Model-Predictive Control (MPC) using the Cross-Entropy Method (CEM) *at every single frame*. 
Let's look at the math of what your GPU is calculating for a *single* step in `03_planet_rssm.py`:
*   `cem_candidates = 1000` (Imagining 1,000 different futures)
*   `planning_horizon = 12` (Looking 12 frames deep)
*   `cem_iters = 10` (Refitting the Gaussian distribution 10 times to find the optimal path)

**Total neural network forward passes per real environment step:** $1000 \times 12 \times 10 = \textbf{120,000 passes}$. 
Your GPU is rendering $120,000$ imaginary latent frames for every 1 true step you take in `envpool`. An SPS of 17 is actually quite impressive for a naive PyTorch implementation without `jax.vmap`!

***

**How long will our run take?**
*   At 17 SPS, generating **$100,000$ steps** (our current script limit) will take about **$1.6 \text{ hours}$**.
*   A full training cycle of **$1,000,000$ steps** would take roughly **$16 \text{ hours}$**.

***

**How many steps and how much time did the original paper take?**

Danijar Hafner's original PlaNet paper (2018) was famous exactly for this tradeoff: **It sacrificed wall-clock time for extreme sample efficiency.**
*   **Total Steps:** The paper established state-of-the-art performance on the DeepMind Control Suite (continuous pixel environments) using only **$500,000 \text{ to } 2,000,000 \text{ environment steps}$** depending on the specific task. (By comparison, standard PPO or DQN would take $10,000,000 \text{ to } 50,000,000$ steps to achieve the same result).
*   **Total Time:** Because of the massively expensive CEM planning step (just like what you are seeing with 17 SPS), the original original TensorFlow implementation took about **$1 \text{ to } 2 \text{ days}$** to complete a 1-million step training run on a relatively powerful **NVIDIA V100 GPU**.

### The Phase 4 Teaser (Dreamer)
This exact computational bottleneck (17 SPS doing 120,000 passes per frame via CEM) is the *primary reason* Phase 4 exists. Next, in **DreamerV1 / DreamerV2**, Hafner removed the CEM planner entirely. Instead of imagining 1,000 paths at runtime, Dreamer trains an Actor-Critic completely *inside* the RSSM's latent imagination offline, giving you the best of Phase 1 and Phase 3 combined: infinite simulated training data, but instant $O(1)$ action selection at test time!




Replacing 2 lines with 79 lines in [](file:///workspace/wiki/research-world-models/phase3_planet_rssm/README.md)


```


###

```

# in readme, detail the entire rssm pipline,
and add code segment to core steps, and explain code to those new to pytorch
i'm confused, why training need to wait MPC? or if it just in inference, why training so slow?


I have updated the `README.md` to directly answer your questions, adding simple, annotated code blocks to demystify the core logic of the RSSM for PyTorch beginners!

### The Training Bottleneck Explained

To directly answer your confusion about why training is so slow even if MPC is just for inference:

In Reinforcement Learning, an agent must **play the game to collect data** to train on. 
Because PlaNet (RSSM) uses CEM for its control policy, it *must* use CEM at **every single environment step** to decide what to do next to actually play the game. 

Since CEM simulates 120,000 forward passes (1000 candidates $\times$ 12 horizon $\times$ 10 iterations) per frame:
The agent takes a massive amount of wall-clock time just to generate a few frames of experience to put into the `ReplayBuffer`. The training backpropagation itself (`loss.backward()`) is actually quite fast! You are essentially waiting for the agent to "think" incredibly deeply before every single move it makes just to generate the `batch_obs` data needed to train the RSSM's KL and Reconstruction loss.

### Code Fragments Added to the README
I have added three very clear sections to the README breaking down the exact PyTorch pipeline:
1. **The RSSM Class**: Explained the difference between the deterministic GRU and the two stochastic layers (Prior vs. Posterior).
2. **The Unroll Loop (Training)**: Showing how the Prior matches the Posterior using `kl_divergence` (Loss Function).
3. **The CEM Planner (Execution)**: Showing how the agent is forced to use *only* the Prior to imagine states during continuous execution without ever seeing the true `envpool` image frame!

Made changes.


# explain more detail, 
- the motivation or theory behind the design of rssm
-  the trainin detail, the loss, the data , how is planner integrate in learning loop?


Replacing 5 lines with 12 lines in [](file:///workspace/wiki/research-world-models/phase3_planet_rssm/README.md)


```

```


Replacing 4 lines with 16 lines in [](file:///workspace/wiki/research-world-models/phase3_planet_rssm/README.md)


```

```
I've updated the `README.md` to directly break down the mathematical motivation behind the RSSM architecture, the training loop dependencies, and explicitly how the three loss functions make the planning possible.

### What Was Expanded in the Documentation:

1.  **The Motivation & Theory (RNNs vs Gaussian Space):** 
    I added a section explaining exactly *why* Hafner split the state into `h` and `s`. 
    *   If you use a pure RNN (like LSTM), it predicts a single point. In stochastic environments where many futures are possible, minimizing MSE makes the RNN output an "average" of all possible futures, causing blurry or impossible predictions.
    *   If you use a pure Gaussian sampling pipeline, the noise over thousands of frames destroys memory.
    *   **The Fix:** Use the deterministic GRU (`h`) to remember the past safely, and append the stochastic Gaussian (`s`) to sample the multiple distinct possibilities of the future.

2.  **The Learning Pipeline (Online MPC):**
    I explained the symbiotic loop:
    1.  The agent uses whatever broken/flawed RSSM it currently has to plan its next action via CEM, saving that result to an **Episodic Replay Buffer**.
    2.  The script then regularly pauses the game, samples 50-frame sequence chunks from the buffer, and unrolls the RSSM through Backpropagation Through Time (BPTT).

3.  **The Losses:**
    This explains exactly *why* it learns to dream:
    *   **Reconstruction (MSE):** Decode the latent state back to an image to ensure it holds real physical data.
    *   **Reward (MSE):** Predict the reward from the latent state.
    *   **KL Divergence:** This is the core magic. The Posterior model is allowed to "cheat" and look at the real image to figure out the state. We force the Prior model (which only knows the past) to mathematically match the distribution of the Posterior. Once the KL Divergence is low, the Prior is so accurate that we can turn off the Posterior completely and just let the Prior dream the future using only actions!

By understanding how the KL Loss forces the blind "Prior" to match the cheating "Posterior," you understand exactly how the CEM Planner is capable of simulating the future without seeing frames.

Made changes.