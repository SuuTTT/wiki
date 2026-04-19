import numpy as np
import torch

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: torch.device = torch.device("cpu"),
        handle_timeout_termination: bool = False,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.handle_timeout_termination = handle_timeout_termination

        self.obs = np.zeros((self.buffer_size,) + self.observation_space.shape, dtype=self.observation_space.dtype)
        self.next_obs = np.zeros((self.buffer_size,) + self.observation_space.shape, dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size,) + self.action_space.shape, dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done, infos):
        # We assume batch size 1 for simplicity in these tutorials, or handle vectorized envs.
        # CleanRL typically handles vectorized envs in the buffer.
        
        # Determine number of envs
        num_envs = obs.shape[0] if len(obs.shape) > len(self.observation_space.shape) else 1
        
        if num_envs == 1 and not isinstance(obs, np.ndarray) or (isinstance(obs, np.ndarray) and len(obs.shape) == len(self.observation_space.shape)):
            self.obs[self.pos] = np.array(obs).copy()
            self.next_obs[self.pos] = np.array(next_obs).copy()
            self.actions[self.pos] = np.array(action).copy()
            self.rewards[self.pos] = np.array(reward).copy()
            self.dones[self.pos] = np.array(done).copy()
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
        else:
            for i in range(num_envs):
                self.obs[self.pos] = np.array(obs[i]).copy()
                self.next_obs[self.pos] = np.array(next_obs[i]).copy()
                self.actions[self.pos] = np.array(action[i]).copy()
                self.rewards[self.pos] = np.array(reward[i]).copy()
                self.dones[self.pos] = np.array(done[i]).copy()
                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
                    self.pos = 0

    def sample(self, batch_size: int):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        # Ensure actions have a shape of (batch_size, 1) for .gather(1, actions)
        actions = self.actions[batch_inds]
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)

        return ReplayBufferSamples(
            observations=torch.Tensor(self.obs[batch_inds]).to(self.device),
            next_observations=torch.Tensor(self.next_obs[batch_inds]).to(self.device),
            actions=torch.LongTensor(actions).to(self.device),
            rewards=torch.Tensor(self.rewards[batch_inds]).to(self.device),
            dones=torch.Tensor(self.dones[batch_inds]).to(self.device),
        )

from dataclasses import dataclass

@dataclass
class ReplayBufferSamples:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
