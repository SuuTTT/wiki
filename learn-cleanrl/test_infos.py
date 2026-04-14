import gymnasium as gym
envs = gym.vector.SyncVectorEnv([
    lambda: gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
])
obs, _ = envs.reset()
for i in range(1000):
    obs, r, t, trunc, infos = envs.step(envs.action_space.sample())
    if t[0] or trunc[0]:
        print("Done:", infos)
        break
