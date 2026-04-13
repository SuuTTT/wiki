import envpool
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# -----------------
# 1. Configuration
# -----------------
ENV_ID = "Pong-v5"
NUM_ENVS = 8
NUM_STEPS = 128
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES
NUM_ITERATIONS = TOTAL_TIMESTEPS // BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------
# 2. EnvPool Vectorized Environment Setup
# -----------------
# Unlike Gym VectorEnvs, envpool runs via a high-performance C++ thread pool.
# We skip the heavy multiprocessing context switching.
envs = envpool.make(
    ENV_ID,
    env_type="gym",
    num_envs=NUM_ENVS,
    episodic_life=True,  # Built-in wrappers
    reward_clip=True,
    seed=1
)

# Standardize names for the tutorial (EnvPool sets these slightly differently)
envs.single_action_space = envs.action_space
envs.single_observation_space = envs.observation_space
envs.num_envs = NUM_ENVS


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Standard CNN Architecture for Atari
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

# -------------
# 3. Buffer Storage Setup
# -------------
obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

start_time = time.time()
global_step = 0

next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(NUM_ENVS).to(device)


for iteration in range(1, NUM_ITERATIONS + 1):
    
    # Anneal Learning Rate
    frac = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
    optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

    for step in range(0, NUM_STEPS):
        global_step += 1 * NUM_ENVS
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # ENVPOOL Execution: envpool returns a batched float array directly.
        # It's zero-copy memory native in C++
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        
        # Logging Episode Lengths/Rewards from the flattened 'info' dictionary
        # In multi-agent envs/Gym, 'info' is usually a List[Dict]. EnvPool returns Dict[BatchArray].
        # For instance, if any environment finished an episode:
        # Envpool adds "reward" natively inside `info` representing cumulative episodic return 
        if "reward" in info:
            # We must iterate over environments that actually reached done
            for i, d in enumerate(done):
                if d and not info.get("TimeLimit.truncated", [False]*NUM_ENVS)[i]: 
                    print(f"global_step={global_step}, ep_return={info['reward'][i]}")

        # Move to GPU
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

    # ----------------
    # 4. GAE & Learning
    # ----------------
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + values

    # Flatten
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # PPO Implementation (Optimization Loop)
    b_inds = np.arange(BATCH_SIZE)
    
    for epoch in range(UPDATE_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy Loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.1, 1 + 0.1)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value Loss
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # Entropy Loss
            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    curr_sps = int(global_step / (time.time() - start_time))
    if iteration % 5 == 0:
        print(f"Iteration {iteration}/{NUM_ITERATIONS} - SPS: {curr_sps}")

print("✅ EnvPool PPO Tutorial Completed!")
envs.close()
