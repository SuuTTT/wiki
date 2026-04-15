# ==============================================================================
# DREAMER V1 TUTORIAL
# Paper: "Dream to Control: Learning Behaviors by Latent Imagination"
#        Hafner et al., 2019  (https://arxiv.org/abs/1912.01603)
# Reference Implementation: https://github.com/danijar/dreamer
# ==============================================================================
#
# OVERVIEW: DreamerV1 builds on PlaNet's RSSM world model but replaces the
# slow CEM planner with a differentiable Actor-Critic trained *purely inside
# the imagined futures* of the world model. The full algorithm has three phases:
#
#   Phase 1 — WORLD MODEL UPDATE
#             Fit the RSSM, image decoder, and reward predictor to real data.
#             Loss = KL(posterior ‖ prior)  +  -log p(image)  +  -log p(reward)
#
#   Phase 2 — ACTOR UPDATE  (no real environment interaction!)
#             Unroll H imagined steps using the RSSM prior + the current actor.
#             Compute λ-returns over the imagined trajectory.
#             Loss = -mean( V_λ )       (actor maximises expected return)
#
#   Phase 3 — CRITIC UPDATE
#             Regress imagined value predictions onto the λ-returns.
#             Loss = MSE( v(z_t), V_λ(z_t) )
#
# ─────────────────────────────────────────────────────────────────────────────
# THE RSSM STATE (Recurrent State Space Model)
# ─────────────────────────────────────────────────────────────────────────────
#   h_t  = GRU( img1(s_{t-1}, a_{t-1}),  h_{t-1} )   ← deterministic path
#
#   PRIOR:      p(s_t | h_t )             ← transition model (no observation)
#   POSTERIOR:  q(s_t | h_t, e_t )        ← conditioned on encoder embedding
#
#   FEATURE:    z_t = concat( h_t, s_t )  ← input to all decoders and policy
#
# The posterior is used for TRAINING (it sees the real observation).
# The prior    is used for IMAGINATION (only uses dynamics, no real obs).
# ==============================================================================

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.kl import kl_divergence
import tyro
import tqdm

import gymnasium as gym
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../learn-cleanrl")))
from cleanrl_utils.logger import RLTracker


# ==============================================================================
# HYPERPARAMETERS  (matching DreamerV1 paper defaults)
# ==============================================================================
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True

    # Environment
    env_id: str = "CarRacing-v3"    # 96×96 pixel obs, 3-D continuous actions
    total_steps: int = 1_000_000
    action_repeat: int = 2          # Repeat each action N times — paper uses 2

    # World Model Architecture
    cnn_depth: int = 32             # Base CNN channels (layers: 1×,2×,4×,8× multiples)
    deter_size: int = 200           # GRU hidden state — deterministic part of RSSM
    stoch_size: int = 30            # Gaussian latent state dimensionality
    hidden_size: int = 200          # Dense width inside RSSM transition heads
    num_units: int = 400            # Dense width for reward / value / actor MLPs

    # Training schedule
    prefill_steps: int = 5000       # Random env steps before any gradient updates
    train_every: int = 1000         # Collect N env steps, then do one training cycle
    train_steps: int = 100          # Gradient steps per training cycle
    batch_size: int = 50            # Sequences per mini-batch
    seq_len: int = 50               # BPTT sequence length
    max_episodes: int = 1000        # Max replay buffer capacity (in episodes)

    # Loss coefficients
    kl_scale: float = 1.0           # Weight on the KL divergence term
    free_nats: float = 3.0          # KL clipped to max(free_nats, KL) — prevents collapse

    # Optimisers
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    grad_clip: float = 100.0

    # Actor-Critic inside imagination
    horizon: int = 15               # Imagination rollout length H
    gamma: float = 0.99             # Discount factor
    lambda_: float = 0.95           # λ for TD(λ) returns

    # Exploration: additive Gaussian noise on actions during collection
    expl_amount: float = 0.3

    capture_video: bool = False


# ==============================================================================
# ENVIRONMENT WRAPPERS
# ==============================================================================

class ResizeObsWrapper(gym.ObservationWrapper):
    """Resize (H, W, C) uint8 pixel obs to (3, 64, 64) using PIL bilinear."""
    def __init__(self, env, size: int = 64):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, size, size), dtype=np.uint8,
        )

    def observation(self, obs):
        img = Image.fromarray(obs).resize((self.size, self.size), Image.BILINEAR)
        return np.array(img, dtype=np.uint8).transpose(2, 0, 1)   # (C,H,W)


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat the same action for `repeat` consecutive env steps, sum rewards."""
    def __init__(self, env, repeat: int):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(args: Args, capture_video: bool = False, run_name: str = "") -> gym.Env:
    """Build one CarRacing-v3 environment with pixel preprocessing."""
    env = gym.make(args.env_id, render_mode="rgb_array")
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", lambda ep: ep % 50 == 0)
    env = ActionRepeatWrapper(env, args.action_repeat)
    env = ResizeObsWrapper(env)                 # → (3, 64, 64) uint8
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)
    return env


# ==============================================================================
# HELPER — PREPROCESS OBSERVATIONS
# ==============================================================================

def preprocess(obs: torch.Tensor) -> torch.Tensor:
    """
    Normalise uint8 pixel observations to float32 in [-0.5, 0.5].
    Matches the original DreamerV1 preprocessing.
    obs: (..., C, H, W) uint8  →  float32 in [-0.5, 0.5]
    """
    return obs.float() / 255.0 - 0.5


# ==============================================================================
# REPLAY BUFFER (episode-based)
# ==============================================================================

class EpisodeBuffer:
    """
    Stores complete episodes as dicts of numpy arrays.
    Sampling draws random contiguous sub-sequences that never cross episode
    boundaries — crucial for the RSSM's temporal consistency.
    """
    def __init__(self, max_episodes: int):
        self.episodes: deque = deque(maxlen=max_episodes)

    def add_episode(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """
        obs     : (T+1, C, H, W) uint8  — the final next_obs is included
        actions : (T,  action_dim)  float32
        rewards : (T,)              float32
        """
        self.episodes.append({"obs": obs, "actions": actions, "rewards": rewards})

    def __len__(self):
        return len(self.episodes)

    def sample(self, batch_size: int, seq_len: int):
        """
        Returns (obs, actions, rewards) as numpy arrays with shapes
          (B, T, C,H,W),  (B, T, A),  (B, T).
        """
        batch_obs, batch_act, batch_rew = [], [], []
        episodes = list(self.episodes)
        while len(batch_obs) < batch_size:
            ep = random.choice(episodes)
            T  = len(ep["rewards"])
            if T < seq_len:
                continue
            t = random.randint(0, T - seq_len)
            batch_obs.append(ep["obs"][t : t + seq_len])
            batch_act.append(ep["actions"][t : t + seq_len])
            batch_rew.append(ep["rewards"][t : t + seq_len])
        return (
            np.stack(batch_obs, axis=0),
            np.stack(batch_act, axis=0),
            np.stack(batch_rew, axis=0),
        )


# ==============================================================================
# NEURAL NETWORK MODULES
# ==============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialisation — good default for deep RL networks."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# CONVOLUTIONAL ENCODER
# ─────────────────────────────────────────────────────────────────────────────
#  Maps 64×64 RGB frames → 1024-dim embedding  (with default depth=32).
#
#  Spatial sizes (depth=32, kernel=4, stride=2, no padding):
#    Input:   (B,  3, 64, 64)
#    Conv1:   (B, 32, 31, 31)   floor((64-4)/2)+1 = 31
#    Conv2:   (B, 64, 14, 14)   floor((31-4)/2)+1 = 14
#    Conv3:   (B,128,  6,  6)   floor((14-4)/2)+1 =  6
#    Conv4:   (B,256,  2,  2)   floor(( 6-4)/2)+1 =  2
#    Flatten: (B, 1024)         2×2×256 = 1024
# ─────────────────────────────────────────────────────────────────────────────
class ConvEncoder(nn.Module):
    def __init__(self, depth: int = 32):
        super().__init__()
        self.embed_dim = 8 * depth * 4   # 8×depth channels × 2×2 spatial = 1024
        self.net = nn.Sequential(
            nn.Conv2d(3,         1*depth, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(1*depth,   2*depth, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(2*depth,   4*depth, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(4*depth,   8*depth, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (..., C, H, W)  — leading dims are collapsed for the convolution.
        Returns: (..., embed_dim)
        """
        shape = obs.shape
        x  = self.net(obs.reshape(-1, *shape[-3:]))
        return x.reshape(*shape[:-3], self.embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# CONVOLUTIONAL DECODER
# ─────────────────────────────────────────────────────────────────────────────
#  Maps the RSSM feature z_t=(h_t,s_t) back to reconstructed 64×64 frames.
#  Output: pixel means (we use MSE loss, equivalent to NLL with fixed σ=1).
#
#  Spatial reconstruction (depth=32, no padding):
#    Project: (B, feature_dim) → (B, 1024, 1, 1)   [32*depth = 1024]
#    CT1 depth=32*d→4d, kernel=5, stride=2:  1 →  5   (1-1)*2+5=5
#    CT2 depth= 4d→2d, kernel=5, stride=2:  5 → 13   (5-1)*2+5=13
#    CT3 depth= 2d→1d, kernel=6, stride=2: 13 → 30   (13-1)*2+6=30
#    CT4 depth= 1d→ 3, kernel=6, stride=2: 30 → 64   (30-1)*2+6=64  ✓
# ─────────────────────────────────────────────────────────────────────────────
class ConvDecoder(nn.Module):
    def __init__(self, feature_dim: int, depth: int = 32):
        super().__init__()
        self.depth = depth
        # Project feature to 32*depth = 1024 channels for the deconv head
        self.fc = nn.Linear(feature_dim, 32 * depth)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32*depth, 4*depth, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d( 4*depth, 2*depth, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d( 2*depth, 1*depth, kernel_size=6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d( 1*depth, 3,       kernel_size=6, stride=2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (..., feature_dim)
        Returns:  (..., 3, 64, 64)  — reconstructed pixel means in approx [-0.5, 0.5]
        """
        shape = features.shape
        x = self.fc(features.reshape(-1, shape[-1]))    # (N, 32*depth)
        x = x.reshape(-1, 32 * self.depth, 1, 1)        # (N, 1024, 1, 1)
        x = self.net(x)                                  # (N, 3, 64, 64)
        return x.reshape(*shape[:-1], 3, 64, 64)


# ─────────────────────────────────────────────────────────────────────────────
# DENSE MLP DECODER — shared backbone for reward and value networks
# ─────────────────────────────────────────────────────────────────────────────
class DenseDecoder(nn.Module):
    """Stack of `layers` Dense→ELU blocks followed by a linear output head."""
    def __init__(self, feature_dim: int, output_dim: int = 1,
                 layers: int = 2, units: int = 400):
        super().__init__()
        dims = [feature_dim] + [units] * layers + [output_dim]
        net  = []
        for i in range(len(dims) - 2):
            net += [layer_init(nn.Linear(dims[i], dims[i + 1])), nn.ELU()]
        net.append(layer_init(nn.Linear(dims[-2], dims[-1]), std=1.0))
        self.net = nn.Sequential(*net)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION MODEL — Tanh-squashed Normal ("tanh_normal") policy
# ─────────────────────────────────────────────────────────────────────────────
#
#  The action distribution is a Normal in pre-tanh space, squashed to (−1,1):
#
#    μ  = mean_scale · tanh(raw_μ / mean_scale)     ← keeps μ bounded in ±5
#    σ  = softplus(raw_σ + raw_init_std) + min_std  ← ≈ init_std=5 at init
#    a  = tanh( Normal(μ, σ).sample() )             ← bounded action
#
#  We use rsample() (reparameterisation trick) so gradients flow backwards
#  through the stochastic action into the actor weights.
#
#  KEY: raw_init_std = log(exp(init_std)−1) — softplus⁻¹ of init_std.
#       This ensures σ ≈ 5 at initialisation → maximally stochastic exploration.
# ─────────────────────────────────────────────────────────────────────────────
class ActionModel(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int,
                 layers: int = 4, units: int = 400,
                 init_std: float = 5.0, min_std: float = 1e-4,
                 mean_scale: float = 5.0):
        super().__init__()
        self.action_dim   = action_dim
        self.min_std      = min_std
        self.mean_scale   = mean_scale
        self.raw_init_std = float(np.log(np.exp(init_std) - 1.0))

        dims = [feature_dim] + [units] * layers
        net  = []
        for i in range(len(dims) - 1):
            net += [layer_init(nn.Linear(dims[i], dims[i + 1])), nn.ELU()]
        self.trunk     = nn.Sequential(*net)
        self.mean_head = layer_init(nn.Linear(units, action_dim), std=0.01)
        self.std_head  = layer_init(nn.Linear(units, action_dim), std=0.01)

    def get_dist(self, features: torch.Tensor) -> TransformedDistribution:
        """Return a tanh-squashed Normal distribution over actions."""
        x    = self.trunk(features)
        mean = self.mean_head(x)
        std  = self.std_head(x)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std  = F.softplus(std + self.raw_init_std) + self.min_std
        # TanhTransform with cache_size=1 required for log_prob after rsample
        return TransformedDistribution(Normal(mean, std), [TanhTransform(cache_size=1)])

    def forward(self, features: torch.Tensor,
                deterministic: bool = False) -> torch.Tensor:
        """
        Sample an action.  Uses rsample() (reparameterised) during training
        so that gradients flow back through the stochastic sample.
        In deterministic mode (evaluation) returns tanh(mean).
        """
        if deterministic:
            x    = self.trunk(features)
            raw  = self.mean_head(x)
            mean = self.mean_scale * torch.tanh(raw / self.mean_scale)
            return torch.tanh(mean)
        return self.get_dist(features).rsample()   # differentiable!


# ─────────────────────────────────────────────────────────────────────────────
# RSSM — Recurrent State Space Model  (the world model core)
# ─────────────────────────────────────────────────────────────────────────────
#
#  State dict keys: {'mean', 'std', 'stoch', 'deter'}
#  • deter  h_t  : deterministic, flows through GRU — long-range memory
#  • stoch  s_t  : stochastic sample from a diagonal Gaussian — compactness
#
#  img_step  (PRIOR — advance without observation):
#    x   = ELU( img1( [s_{t-1} ‖ a_{t-1}] ) )     Dense hidden_size
#    h_t = GRU(x, h_{t-1})                          GRU deter_size
#    x   = ELU( img2(h_t) )                         Dense hidden_size
#    μ,σ = split( img3(x) )                         Dense→2×stoch_size
#    s_t ~ Normal(μ, softplus(σ)+0.1)
#
#  obs_step  (POSTERIOR — incorporate real observation):
#    prior           = img_step(prev, a_{t-1})
#    x   = ELU( obs1( [prior.deter ‖ embed_t] ) )  Dense hidden_size
#    μ,σ = split( obs2(x) )                         Dense→2×stoch_size
#    s_t ~ Normal(μ, softplus(σ)+0.1)
# ─────────────────────────────────────────────────────────────────────────────
class RSSM(nn.Module):
    def __init__(self, action_dim: int,
                 stoch_size: int = 30,
                 deter_size: int = 200,
                 hidden_size: int = 200,
                 embed_dim: int = 1024):
        super().__init__()
        self.stoch_size  = stoch_size
        self.deter_size  = deter_size

        # Prior path (img = "imagine")
        self.img1 = layer_init(nn.Linear(stoch_size + action_dim, hidden_size))
        self.gru  = nn.GRUCell(hidden_size, deter_size)
        self.img2 = layer_init(nn.Linear(deter_size,  hidden_size))
        self.img3 = layer_init(nn.Linear(hidden_size, 2 * stoch_size))

        # Posterior correction path (obs = "observe")
        self.obs1 = layer_init(nn.Linear(deter_size + embed_dim, hidden_size))
        self.obs2 = layer_init(nn.Linear(hidden_size, 2 * stoch_size))

    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        return {
            "mean":  torch.zeros(batch_size, self.stoch_size, device=device),
            "std":   torch.zeros(batch_size, self.stoch_size, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_size, device=device),
            "deter": torch.zeros(batch_size, self.deter_size, device=device),
        }

    def get_feat(self, state: dict) -> torch.Tensor:
        """Feature vector z_t = concat(h_t, s_t)."""
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    # ── single-step ──────────────────────────────────────────────────────────

    def img_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        """
        PRIOR step — advance one step using only dynamics (no observation).
        Used during imagination rollouts.
        """
        x     = F.elu(self.img1(torch.cat([prev_state["stoch"], prev_action], dim=-1)))
        deter = self.gru(x, prev_state["deter"])
        x     = F.elu(self.img2(deter))
        x     = self.img3(x)
        mean, std_raw = x.chunk(2, dim=-1)
        std   = F.softplus(std_raw) + 0.1
        stoch = mean + std * torch.randn_like(std)   # reparameterised
        return {"mean": mean, "std": std, "stoch": stoch, "deter": deter}

    def obs_step(self, prev_state: dict, prev_action: torch.Tensor,
                 embed: torch.Tensor) -> tuple:
        """
        POSTERIOR step — update RSSM belief using an encoder embedding.
        Returns (posterior_state, prior_state) — both needed for the KL loss.
        """
        prior = self.img_step(prev_state, prev_action)
        x     = F.elu(self.obs1(torch.cat([prior["deter"], embed], dim=-1)))
        x     = self.obs2(x)
        mean, std_raw = x.chunk(2, dim=-1)
        std   = F.softplus(std_raw) + 0.1
        stoch = mean + std * torch.randn_like(std)
        post  = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    # ── sequence rollouts ────────────────────────────────────────────────────

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor,
                init: dict = None) -> tuple:
        """
        Run RSSM over a full sequence using posteriors (real data).

        embeds  : (B, T, embed_dim)
        actions : (B, T, action_dim)  — a_t taken AFTER observing o_t
        Returns : (posts, priors), each a dict of (B, T, dim) tensors
        """
        B, T   = actions.shape[:2]
        device = actions.device
        state  = init if init is not None else self.initial_state(B, device)

        posts, priors = [], []
        prev_action   = torch.zeros(B, actions.shape[-1], device=device)

        for t in range(T):
            post, prior = self.obs_step(state, prev_action, embeds[:, t])
            posts.append(post)
            priors.append(prior)
            state       = post
            prev_action = actions[:, t]   # action taken FROM this posterior state

        posts  = {k: torch.stack([p[k] for p in posts],  dim=1) for k in posts[0]}
        priors = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0]}
        return posts, priors

    def imagine(self, actor: "ActionModel", init: dict, horizon: int) -> dict:
        """
        Roll out H imagined steps using the PRIOR (no real observations).
        The actor provides differentiable actions via rsample() so gradients
        can propagate backwards through the entire imagined trajectory.

        init     : state dict, each value shape (N, dim)  where N = B*T flattened
        Returns  : dict of (H, N, dim) stacked imagined states
        """
        state = init
        states_list = []
        for _ in range(horizon):
            feat   = self.get_feat(state)
            action = actor(feat)          # differentiable tanh-Normal sample
            state  = self.img_step(state, action)
            states_list.append(state)
        return {k: torch.stack([s[k] for s in states_list], dim=0) for k in states_list[0]}


# ==============================================================================
# LAMBDA RETURNS  (TD(λ))
# ==============================================================================
def compute_lambda_returns(rewards: torch.Tensor, values: torch.Tensor,
                           gamma: float, lambda_: float) -> torch.Tensor:
    """
    Compute λ-returns (the training targets) over an imagined trajectory.

    MATH — recursive definition (backwards scan):
      V_λ(H-1) = r_{H-1} + γ · v_H           ← pure TD(0) bootstrap at horizon end
      V_λ(t)   = r_t + γ [ (1-λ) v(t+1) + λ V_λ(t+1) ]   for t < H-1

    Intuition:
      λ=0 → pure TD(0): V_λ(t) = r_t + γ v(t+1)   (low variance, high bias)
      λ=1 → Monte Carlo: V_λ(t) = Σ γ^k r_{t+k}    (high variance, low bias)
      λ=0.95 (paper default) → smooth bias-variance tradeoff

    rewards : (H, N)   — imagined rewards  (H steps total)
    values  : (H, N)   — value estimates   (values[-1] is the bootstrap)
    Returns : (H-1, N) — λ-returns for the first H-1 imagined steps
    """
    H        = rewards.shape[0]
    last     = values[H - 1]            # bootstrap from the value at step H
    returns  = []

    for t in reversed(range(H - 1)):
        v_next = values[t + 1] if t < H - 2 else values[H - 1]
        last   = rewards[t] + gamma * ((1.0 - lambda_) * v_next + lambda_ * last)
        returns.insert(0, last)

    return torch.stack(returns, dim=0)  # (H-1, N)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    args = tyro.cli(Args)

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    tracker  = RLTracker(args.exp_name, args.seed)

    # ── Environment ───────────────────────────────────────────────────────────
    env          = make_env(args, capture_video=args.capture_video, run_name=run_name)
    action_dim   = int(np.prod(env.action_space.shape))
    print(f"Env: {args.env_id}  obs={env.observation_space.shape}  action_dim={action_dim}")

    # ── Build models ──────────────────────────────────────────────────────────
    encoder      = ConvEncoder(depth=args.cnn_depth).to(device)
    embed_dim    = encoder.embed_dim                              # 1024 with depth=32
    feature_dim  = args.deter_size + args.stoch_size              # z_t = (h_t, s_t)

    rssm         = RSSM(action_dim, args.stoch_size, args.deter_size,
                        args.hidden_size, embed_dim).to(device)
    decoder      = ConvDecoder(feature_dim, depth=args.cnn_depth).to(device)
    reward_model = DenseDecoder(feature_dim, output_dim=1, layers=2,
                                units=args.num_units).to(device)
    value_model  = DenseDecoder(feature_dim, output_dim=1, layers=3,
                                units=args.num_units).to(device)
    actor_model  = ActionModel(feature_dim, action_dim, layers=4,
                               units=args.num_units).to(device)

    # ── Optimisers ────────────────────────────────────────────────────────────
    model_params = (list(encoder.parameters())      +
                    list(rssm.parameters())          +
                    list(decoder.parameters())       +
                    list(reward_model.parameters()))
    optim_model  = optim.Adam(model_params,              lr=args.model_lr)
    optim_actor  = optim.Adam(actor_model.parameters(),  lr=args.actor_lr)
    optim_value  = optim.Adam(value_model.parameters(),  lr=args.value_lr)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    replay = EpisodeBuffer(max_episodes=args.max_episodes)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 0 — PREFILL REPLAY BUFFER WITH RANDOM ACTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    #  Before any model training can start we need enough diverse data.
    #  Collect complete episodes using random actions and store them.
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\nPrefilling replay buffer with {args.prefill_steps} random steps …")
    prefill_steps = 0
    while prefill_steps < args.prefill_steps:
        obs_np, _ = env.reset()
        ep_obs, ep_act, ep_rew = [obs_np], [], []
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs_np, reward, terminated, truncated, _ = env.step(action)
            ep_obs.append(next_obs_np)
            ep_act.append(action)
            ep_rew.append(reward)
            done          = terminated or truncated
            obs_np        = next_obs_np
            prefill_steps += 1
        replay.add_episode(
            np.stack(ep_obs).astype(np.uint8),
            np.stack(ep_act).astype(np.float32),
            np.array(ep_rew, dtype=np.float32),
        )
    print(f"Replay buffer: {len(replay)} episodes after prefill.")

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    global_step        = prefill_steps
    steps_since_train  = 0
    obs_np, _          = env.reset()
    ep_obs             = [obs_np]
    ep_act, ep_rew     = [], []

    # Per-episode RSSM state — reset when episode ends
    h_t = torch.zeros(1, args.deter_size, device=device)
    s_t = torch.zeros(1, args.stoch_size,  device=device)
    prev_action_online = torch.zeros(1, action_dim, device=device)  # track actual prev action
    last_ep_ret = float('nan')  # for display between episodes

    pbar = tqdm.tqdm(total=args.total_steps, initial=global_step,
                     desc="DreamerV1", unit="step")

    while global_step < args.total_steps:

        # ── 1. ENVIRONMENT STEP ───────────────────────────────────────────────
        #  Encode the current obs, do a posterior RSSM update, sample an action,
        #  add exploration noise, then step the environment.
        # ─────────────────────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t = preprocess(
                torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)
            )                                                       # (1, C, H, W)
            embed = encoder(obs_t)                                  # (1, embed_dim)

            # Posterior update: incorporate the actual observation into the belief
            prev_state = {"mean": torch.zeros_like(s_t), "std": torch.zeros_like(s_t),
                          "stoch": s_t, "deter": h_t}
            post, _    = rssm.obs_step(
                prev_state,
                prev_action_online,   # actual previous action (not dummy zeros)
                embed,
            )
            h_t, s_t = post["deter"], post["stoch"]

            feat      = rssm.get_feat(post)                        # (1, feature_dim)
            action    = actor_model(feat)                           # (1, action_dim)

            # EXPLORATION: additive Gaussian noise (σ=0.3, clipped to valid range)
            action = torch.clamp(
                action + args.expl_amount * torch.randn_like(action), -1.0, 1.0
            )
            prev_action_online = action  # save for next RSSM step
            action_np = action.cpu().numpy()[0]

        next_obs_np, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        ep_obs.append(next_obs_np)
        ep_act.append(action_np)
        ep_rew.append(float(reward))
        obs_np = next_obs_np

        global_step       += 1
        steps_since_train += 1
        tracker.step(1)
        pbar.update(1)
        # Update tqdm display every step (not just on episode end)
        if global_step % 100 == 0:
            pbar.set_postfix({"ep_ret": f"{last_ep_ret:.1f}", "buf_eps": len(replay)})

        if done:
            ep_return = float(sum(ep_rew))
            ep_length = len(ep_rew)
            last_ep_ret = ep_return
            tracker.log_episode(ep_return, ep_length)
            pbar.set_postfix({"ep_ret": f"{ep_return:.1f}", "buf_eps": len(replay)})

            replay.add_episode(
                np.stack(ep_obs).astype(np.uint8),
                np.stack(ep_act).astype(np.float32),
                np.array(ep_rew, dtype=np.float32),
            )
            obs_np, _ = env.reset()
            ep_obs    = [obs_np]
            ep_act, ep_rew = [], []
            h_t = torch.zeros(1, args.deter_size, device=device)
            s_t = torch.zeros(1, args.stoch_size,  device=device)
            prev_action_online = torch.zeros(1, action_dim, device=device)

        # ── 2. TRAINING CYCLE ─────────────────────────────────────────────────
        #  Every train_every env steps: run train_steps gradient updates.
        # ─────────────────────────────────────────────────────────────────────
        if steps_since_train < args.train_every or len(replay) < 2:
            continue
        steps_since_train = 0

        for _ in range(args.train_steps):

            # Sample a batch of (B, T) sequences from the episode replay buffer
            b_obs_np, b_act_np, b_rew_np = replay.sample(args.batch_size, args.seq_len)

            b_obs = preprocess(
                torch.tensor(b_obs_np, dtype=torch.float32).to(device)
            )                          # (B, T, 3, 64, 64)
            b_act = torch.tensor(b_act_np, dtype=torch.float32).to(device)  # (B, T, A)
            b_rew = torch.tensor(b_rew_np, dtype=torch.float32).to(device)  # (B, T)

            B, T = b_act.shape[:2]

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1 — WORLD MODEL UPDATE
            # ═══════════════════════════════════════════════════════════════════
            #
            #  KL LOSS
            #  ────────
            #  We train the PRIOR p(s_t|h_t) to match the POSTERIOR q(s_t|h_t,o_t).
            #  The posterior has access to the real obs; the prior must predict
            #  without it.  Minimising KL:
            #    • Pulls the prior distribution closer to the posterior (better dynamics)
            #    • The prior becomes a compressed predictive model of the world
            #
            #  Free nats: max(free_nats, KL) — allows KL < 3 nats without penalty,
            #  giving the posterior slack to encode details that vary unpredictably.
            #  This prevents the posterior from collapsing to the prior too early.
            #
            #  IMAGE LOSS
            #  ──────────
            #  MSE(decoder(z_t), normalised_obs_t) ≡ -log p(o_t|z_t) with σ=1.
            #  Forces the latent state z_t to contain enough information to
            #  reconstruct the input frame.
            #
            #  REWARD LOSS
            #  ────────────
            #  MSE(reward_model(z_t), r_t).  Ensures z_t encodes task-relevant info.
            # ═══════════════════════════════════════════════════════════════════

            # Encode all (B,T) observations in one parallel batch
            embed = encoder(b_obs.reshape(B * T, *b_obs.shape[2:]))       # (B*T, E)
            embed = embed.reshape(B, T, -1)                                # (B, T, E)

            # Run RSSM over the full sequence to get posteriors and priors
            posts, priors = rssm.observe(embed, b_act)
            # posts, priors: each dict of (B, T, dim)

            # Flatten (B,T) → N for decoder/reward/KL computation
            feats = rssm.get_feat(
                {k: v.reshape(B * T, -1) for k, v in posts.items()}
            )                                                              # (N, F)

            recon           = decoder(feats)                               # (N,3,64,64)
            target_imgs     = b_obs.reshape(B * T, *b_obs.shape[2:])      # (N,3,64,64)
            recon_loss      = F.mse_loss(recon, target_imgs)

            # Reward off-by-one fix: feat at time t encodes the transition
            # caused by prev_action (= action_{t-1}), so it should predict
            # reward_{t-1} (the reward observed upon arrival at state t).
            # Skip t=0 (no meaningful prev_action) and use feats[1:] → rewards[:-1]
            all_feats_BT    = torch.cat([posts["deter"], posts["stoch"]], dim=-1)  # (B,T,F)
            reward_feats    = all_feats_BT[:, 1:].reshape(B * (T - 1), -1)  # skip t=0
            pred_reward     = reward_model(reward_feats).squeeze(-1)         # (B*(T-1),)
            reward_loss     = F.mse_loss(pred_reward, b_rew[:, :-1].reshape(B * (T - 1)))

            post_dist       = Normal(posts["mean"],  posts["std"])
            prior_dist      = Normal(priors["mean"], priors["std"])
            kl              = kl_divergence(post_dist, prior_dist).sum(-1) # (B, T)
            kl_raw          = kl.mean()  # for logging (before free_nats clamp)
            kl_loss         = torch.clamp(kl, min=args.free_nats).mean()

            model_loss = args.kl_scale * kl_loss + recon_loss + reward_loss

            optim_model.zero_grad()
            model_loss.backward()
            nn.utils.clip_grad_norm_(model_params, args.grad_clip)
            optim_model.step()

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2 — ACTOR UPDATE  (purely imagined — no real env step!)
            # ═══════════════════════════════════════════════════════════════════
            #
            #  Start from each of the B*T posterior states as independent starting
            #  points.  Unroll H=15 imagined steps using the PRIOR dynamics and
            #  the current actor.  Because every step is differentiable:
            #    • The actor's rsample() samples ∂a/∂θ_actor
            #    • img_step propagates them forward through the GRU
            #  …so the gradient ∂L_actor/∂θ_actor flows back through the full
            #  imagined trajectory — no environment needed!
            #
            #  λ-RETURN  (training signal in imagination):
            #    V_λ(t) = r_t + γ[(1-λ)v(t+1) + λ V_λ(t+1)]
            #  λ=0.95: mixes TD(0) (low variance, high bias) with
            #          Monte Carlo  (high variance, low bias).
            # ═══════════════════════════════════════════════════════════════════
            # Subsample starting states for imagination to improve speed.
            # Using every 4th timestep instead of all B*T gives ~4x speedup on
            # Phase 2/3 with minimal loss of diversity.
            flat_posts = {k: v.reshape(B * T, -1).detach() for k, v in posts.items()}
            imag_indices = torch.randperm(B * T, device=device)[:B * T // 4]
            init_state = {k: v[imag_indices] for k, v in flat_posts.items()}
            N = init_state["deter"].shape[0]

            # Imagine H steps forward (returns dict of (H, N, dim))
            imag_states  = rssm.imagine(actor_model, init_state, args.horizon)
            imag_feats   = torch.cat([imag_states["deter"],
                                      imag_states["stoch"]], dim=-1)  # (H, N, F)

            imag_rewards = reward_model(imag_feats).squeeze(-1)        # (H, N)
            imag_values  = value_model(imag_feats).squeeze(-1).detach()# (H, N) no grad

            # λ-returns over the first H-1 imagined steps (H-1, N)
            returns = compute_lambda_returns(
                imag_rewards, imag_values, args.gamma, args.lambda_
            )

            # Exponential discount weights (step 0 is most trustworthy)
            discount = args.gamma ** torch.arange(
                args.horizon - 1, device=device, dtype=torch.float32
            ).unsqueeze(-1)                                            # (H-1, 1)

            actor_loss = -(discount * returns).mean()

            optim_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            optim_actor.step()

            # ═══════════════════════════════════════════════════════════════════
            # PHASE 3 — VALUE (CRITIC) UPDATE
            # ═══════════════════════════════════════════════════════════════════
            #
            #  Train the critic to accurately predict V_λ at each imagined state.
            #  The targets (λ-returns) and imagined features are both detached —
            #  value gradients must NOT flow back into the RSSM or actor.
            #
            #  V (z_t; θ_v) ← V_λ(t)    via MSE
            # ═══════════════════════════════════════════════════════════════════
            value_feats  = imag_feats[:-1].reshape(-1, feature_dim).detach()  # (H-1)*N
            value_pred   = value_model(value_feats).squeeze(-1)
            value_target = returns.reshape(-1).detach()

            value_loss = F.mse_loss(value_pred, value_target)

            optim_value.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip)
            optim_value.step()

        # ── Log losses ────────────────────────────────────────────────────────
        tracker.log_metrics("losses", {
            "kl":       kl_loss.item(),
            "kl_raw":   kl_raw.item(),
            "recon":    recon_loss.item(),
            "reward":   reward_loss.item(),
            "model":    model_loss.item(),
            "actor":    actor_loss.item(),
            "value":    value_loss.item(),
        })
        tracker.log_sps()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    pbar.close()
    env.close()

    tracker.save_checkpoint({
        "encoder":      encoder.state_dict(),
        "rssm":         rssm.state_dict(),
        "decoder":      decoder.state_dict(),
        "reward_model": reward_model.state_dict(),
        "value_model":  value_model.state_dict(),
        "actor_model":  actor_model.state_dict(),
    })
    tracker.close()
    print("Training complete.")
