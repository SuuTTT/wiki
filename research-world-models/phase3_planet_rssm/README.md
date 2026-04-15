# Phase 3: Recurrent State Space Models (RSSM) vs. World Models

This document breaks down the architectural and methodological paradigms across the three distinct approaches to World Modeling we've explored.

## 1. David Ha's Original World Model (2018)
**"The Offline Dreamer"**
* **Components:** VAE (Vision) + MDN-RNN (Memory) + Linear Controller (Action).
* **State Mapping:** The VAE encodes pixels into a stochastic latent vector $z$. The RNN takes $z$ and actions to deterministically predict the *next* $z$.
* **Training Pipeline (Strictly Sequential & Offline):**
  1. Rollout absolute random policies to gather 10,000 frames. Train VAE on these frames. Freeze VAE.
  2. Map all frames to $z$. Train MDN-RNN to predict $z_{t+1}$ given $z_t$. Freeze RNN.
  3. Evolve a generic Linear Controller using **CMA-ES**. The evolution happens *entirely inside the hallucination* of the RNN. The agent never sees the true environment during policy optimization.
* **Flaw:** The RNN must predict purely stochastic $z$ states. If the environment has deterministic elements and stochastic elements intertwined, standard RNNs struggle to balance both without sequence collapse or exploding variance over long horizons.

## 2. Our Phase 1 Integrated World Model
**"Online Auxiliary Representation Learning"**
* **Components:** CNN Encoder + LSTM + PPO Actor-Critic.
* **State Mapping:** Analogous to Ha's model, but tightly coupled. The CNN extracts features, and the LSTM builds temporal memory.
* **Training Pipeline (Joint PyTorch Backpropagation):**
  1. We threw away CMA-ES and offline freezing.
  2. We trained the VAE reconstruction loss, the RNN transition loss, and the PPO policy/value gradients **simultaneously** in a single backward pass. 
  3. We skipped the "dreaming". The agent interacted directly within `envpool`. 
* **Takeaway:** This isn't true MPC (Model Predictive Control). It's using World Model architecture as an *auxiliary loss* to force the PPO agent to learn highly compressed, temporally-aware state representations.

## 3. PlaNet & The Recurrent State Space Model (RSSM - 2019)
**"MPC in Latent Space"**

### The Motivation and Theory Behind RSSM
Prior to RSSM, researchers faced a dilemma:
* **Purely Deterministic RNNs:** If you use a standard RNN (like LSTM/GRU) to predict the future, it generates a single deterministic point. If the environment is stochastic (multiple possible futures from the same state), the RNN minimizes MSE by averaging the futures, leading to blurry, collapsed predictions.
* **Purely Stochastic Models:** If you use a sequence of purely random Gaussian variables, the model struggles to stably pass information (memory) across hundreds of time steps because sampling noise destroys information over time.

**The RSSM Breakthrough:** Hafner et al. solved this by splitting the latent state into two complementary parts:
  * $h_t$: **Deterministic State** (GRU memory). It integrates past information and passes it reliably across time without sampling noise.
  * $s_t$: **Stochastic State** (Gaussian). It branches out to model the multiple possible, uncertain futures at any specific time step.
By conditioning the stochastic state on the deterministic memory, the model can imagine diverse futures without forgetting the past.
* **Prior vs. Posterior:** 
  * The **Prior** ($p(s_t|h_t)$) dreams the next state *without* seeing the true frame.
  * The **Posterior** ($q(s_t|h_t, x_t)$) updates the belief *after* seeing the true frame.
  * Training forces the Prior to match the Posterior via KL Divergence.
### The Data and The Learning Loop
Unlike Ha's offline model, PlaNet trains online. The planner and the model-training depend on each other:
1. **Interact & Plan (Data Collection):** At state $x_t$, the CEM Planner uses the current, imperfect RSSM to imagine futures and picks an action $a_t$. The real environment transitions to $x_{t+1}$ and yields $r_t$. This transition is saved into an **episodic Replay Buffer**.
2. **Train the Model:** In the background, the system samples contiguous sequences (e.g., 50 frames) from the Replay Buffer.
3. **BPTT (Backpropagation Through Time):** The model is unrolled over these 50 frames to compute gradients for three distinct losses.

### The Three Losses
During the training step, the objective is to make the RSSM an accurate simulator of reality:
* **Reconstruction Loss ($L_{recon}$):** The model decodes the latent state $(h_t, s_t)$ back into an image. MSE against the true $x_t$ ensures the latent space accurately visually represents reality instead of collapsing to trivial zeroes.
* **Reward Loss ($L_{reward}$):** The model predicts the reward $r_t$ from $(h_t, s_t)$. MSE against the true reward ensures the latent space captures "what is good/bad for the task."
* **KL Divergence Loss ($L_{KL}$):** This is the core magic! The "Posterior" network gets to see the true image $x_t$ to form a solid belief about the world, creating the target distribution $s_t$. The "Prior" network must guess $s_t$ blindly, using only the past memory $h_{t-1}$ and action $a_{t-1}$. The KL loss forces the Prior to match the Posterior. If trained successfully, the Prior becomes so accurate at predicting the future that we can use it to play the game without seeing the real images!

### Control Paradigm & Planner Integration (No Actor Network!)
Because the Prior learns to accurately dream the future via the KL loss, PlaNet can use the **CEM (Cross-Entropy Method)** as its controller directly on top of the Prior. There is no PPO actor or evolved CMA-ES controller. 

At every frame, the agent uses the optimized Prior to literally imagine 1,000 different futures. It evaluates the outcomes of those imagined paths using the optimized `RewardPredictor`, and executes the first action of the most profitable timeline. It is true Model Predictive Control happening exclusively inside the latent space parameters.

## 4. Why is PlaNet Training So Slow? (The SPS Bottleneck)

You might be wondering: *"If CEM is just for inference (picking actions), why is the training script so slow?"*

In Reinforcement Learning, **training requires collecting data**. To collect high-quality data to train the RSSM, the agent must play the game. In PlaNet, to play the game, the agent *must* use the CEM Planner at **every single environment step**. 

Because CEM imagines 1,000 different 12-step futures (120,000 neural network forward passes) for *one* single action, gathering 1 batch of 50 frames to train on takes a massive amount of wall-clock time. You aren't just waiting for backpropagation; you are waiting for the agent to "think" incredibly hard before every single move it makes just to generate the `ReplayBuffer` experience.

## 5. The Core RSSM Pipeline in PyTorch

Here is the exact code breakdown of how the Recurrent State Space Model works, annotated for PyTorch beginners.

### The RSSM Class
The RSSM needs to manage both deterministic memory (what happened) and stochastic belief (what might happen).

```python
class RSSM(nn.Module):
    def __init__(self, action_dim, stoch_size, deter_size, hidden_size):
        super().__init__()
        # 1. The Deterministic Memory (GRU)
        # nn.GRUCell takes (stochastic_state + action) and updates deterministic_state
        self.cell = nn.GRUCell(stoch_size + action_dim, deter_size)
        
        # 2. The Prior (Imagination)
        # Predicts the stochastic state using ONLY the deterministic memory
        self.fc_prior_mean = nn.Linear(hidden_size, stoch_size)
        self.fc_prior_std = nn.Linear(hidden_size, stoch_size)
        
        # 3. The Posterior (Observation)
        # Predicts the stochastic state using memory PLUS the real image from the CNN
        self.fc_post_mean = nn.Linear(hidden_size, stoch_size)
        self.fc_post_std = nn.Linear(hidden_size, stoch_size)
```

### The Unroll Loop (Training)
In training, we calculate the KL divergence. We want our "blind" Prior to learn to accurately predict what the "seeing" Posterior knows.

```python
# 'h' is deterministic memory, 's' is stochastic state
for t in range(seq_len):
    # 1. Drive Memory Forward with last state and last action
    h = model.rssm.step_forward(h, s, batch_act[:, t])
    
    # 2. See the True Image
    embed = model.encoder(batch_obs[:, t])
    
    # 3. Calculate Posterior (Observation)
    post_mean, post_std = model.rssm.posterior(h, embed)
    post_dist = Normal(post_mean, post_std)
    s = post_dist.rsample() # Sample with gradients (reparameterization trick)
    
    # 4. Calculate Prior (Imagination)
    prior_mean, prior_std = model.rssm.prior(h)
    prior_dist = Normal(prior_mean, prior_std)
    
    # 5. KL Loss: Force Imagination to match Observation
    kl_loss += kl_divergence(post_dist, prior_dist)
```

### The CEM Planner (Execution)
When playing the game, we *only* use the Prior to imagine the future.

```python
for t in range(planning_horizon):
    # Drive memory forward using imagined actions
    h_t = model.rssm.step_forward(h_t, s_t, actions[t])
    
    # Imagine the next state (Prior ONLY - no access to the true image!)
    prior_mean, prior_std = model.rssm.prior(h_t)
    s_t = prior_mean + prior_std * torch.randn_like(prior_mean)
    
    # Predict the reward of this imagined state
    features = torch.cat([h_t, s_t], dim=-1)
    returns += model.reward_model(features)
```

## 6. Deep Dive: The Data, The Planner, and The RL Paradigm

### Why not just train on random trajectories instead of using the Planner's data?
You *can* start with random data (and we actually do, using random actions for the first 1,000 steps to "warm up" the model). However, if an agent just takes random actions, it will never reach complex, deep, or high-reward parts of the environment (like successfully navigating a twisting race track or balancing a pole). 
If the RSSM never sees these advanced states, it can never learn their visual representations or their physics. Training the NN on data collected *by the Planner* creates a natural targeted curriculum: as the model gets smarter, the planner reaches further into the game, giving the model new, complex transition data to learn from.

### Isn't this an RL paradigm? How does it train a pure Neural Network?
Yes, the loop of "Interact -> Store in Buffer -> Sample Batch -> Train" is the classic Reinforcement Learning paradigm (like DQN or SAC). 
However, the critical difference is *what* is being updated. In pure RL, you update a Policy Network (Actor) to maximize a surrogate reward. In PlaNet, there is **no trainable Policy Network**. The RL-style loop is used purely to feed data to train the **World Model** (the pure Neural Network) using standard Supervised/Unsupervised Learning (Reconstruction MSE, Reward MSE, and KL Divergence). The planner (CEM) has no neural network weights; it is just a fixed mathematical search algorithm running on top of the NN's predictions.

### What exactly is CEM? How does it plan in the latent space?
CEM stands for the **Cross-Entropy Method**, a gradient-free optimization algorithm. Here is how it plans without seeing pixels:
1. **Guess:** Generate $N$ (e.g., 1000) completely random action sequences, 12 steps into the future. **(CRITICAL NOTE: This happens entirely in the neural network's imagination. Zero interaction with the real `envpool` environment occurs here!)**
2. **Imagine:** Pass these 1000 action sequences through the RSSM's *Prior* network. The RSSM dynamically updates its memory ($h_t$) and guesses the next states ($s_t$). **This happens entirely in the compressed latent space**, avoiding the massive computational cost of generating 64x64 pixel images. There is no `env.step()`.
3. **Score:** Pass the imagined latent states $(h_t, s_t)$ through the `RewardPredictor` to score all 1000 timelines.
4. **Refine:** Sort and take the top $K$ (e.g., 100) highest-scoring timelines. Calculate the mean and variance of their actions to create a new, better distribution.
5. **Iterate:** Draw a new set of 1000 actions from this refined distribution and repeat.
Finally, after imagining thousands of futures, the agent executes the first step of the absolute best-found sequence in the **real world** (one single `env.step()`). This is the **only** time the agent actually interacts with the true physics engine!

### When exactly does the RSSM interact with the real environment?
This is a critical point of confusion when learning Model-Based RL. Let's trace the exact boundary between the **Real World** (True Physics & Canvas Rendering) and the **Imagination** (Latent Space Math):

1. **Observe True Reality:** The agent starts at $t=0$. It looks at the real 64x64 pixel screen from the game (`obs`). We encode this into the latent vector $s_0$ using the `Encoder` and the `Posterior`.
2. **Imagine Futures (No Interaction):** Using the CEM Planner, the agent enters a loop. It generates 1,000 random actions ($a_0, a_1, \dots, a_{12}$). It feeds these actions into the `RSSM.prior` to calculate where it *thinks* the car will be ($s_1, s_2, \dots, s_{12}$) and what the reward might be. **Zero interaction with the real game (`envpool`) happens here.** This is pure matrix multiplication on the GPU. Time in the real game is completely paused.
3. **Select Best Path:** The Planner identifies the mathematically optimal imaginary sequence. Let's say the winning imaginary sequence was: [Accelerate, Left, Right, Brake].
4. **Execute in Reality (Interaction!):** The agent takes the **very first action** of the winning imaginary sequence ("Accelerate"). It calls `env.step(Accelerate)`. 
5. **The Real World Moves:** The game engine advances exactly one frame. A new physical state ($x_1$) is rendered. The agent gets a true reward $r_0$.
6. **Repeat:** The agent calculates the new latent state $s_1$ based on this real image, and completely discards the rest of its imaginary plan ([Left, Right, Brake]). It starts the CEM planning process all over again from scratch based on the new reality!

### How to Accelerate RSSM Training? (The Birth of Dreamer)
Because the CEM planner is so brutally slow (calculating 120,000 forward passes per real environment step), researchers desperately needed a way to accelerate RSSM. 

**The Solution exists: Dreamer (Phase 4).**
Danijar Hafner (the same author of PlaNet) realized that if the RSSM's latent space is already a perfect differentiable simulation of reality, you don't need a slow, gradient-free search algorithm like CEM at runtime.
* Instead, Dreamer replaces CEM with a standard **Actor-Critic Policy Network** (like PPO/SAC).
* **The Trick:** The Actor-Critic is trained **entirely offline** by backpropagating through the RSSM's imagined latent sequences! You allow the Actor to literally dream millions of frames in a fraction of a second on the GPU, updating its policy using PyTorch gradients.
* **The Result:** At inference time in the real world, the Dreamer agent just passes the image through the Encoder, gets the latent state, and the Actor instantly spits out 1 single optimal action in $O(1)$ time. Training speed goes through the roof, and the SPS bottleneck is eliminated.

### Can RSSM reconstruct the original pixels, states, and dynamics?
**Yes.** 
* **The Pixels:** We have a dedicated `Decoder` network. Its entire job is to take the latent state $(h_t, s_t)$ and deconvolve it back into the original 64x64 pixel image. The Reconstruction Loss mathematically forces the latent vectors to retain precise spatial and visual data.
* **The Dynamics:** The transition elements (`step_forward` GRU + `prior`) force the network to learn the **dynamics** (laws of physics). By enforcing the KL Divergence, the model learns exactly how actions affect the world across time. 
Because the RSSM reconstructs both vision and physics accurately, it functions exactly as a perfectly compressed simulation of reality.

