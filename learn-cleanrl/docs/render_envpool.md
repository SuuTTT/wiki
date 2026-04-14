# Rendering EnvPool Models

When training with EnvPool, video generation is disabled during the high-speed training loop because Python-based rendering (like `gymnasium.wrappers.RecordVideo`) would completely destroy the C++ thread-pool performance gains.

However, it **is** completely possible (and common practice) to render the learned policy *after* training is complete. The standard procedure is to save the PyTorch model weights to disk, and then build a separate "enjoy/evaluation" script.

## The Two-Step Process

1. **Save Model Weights:** At the end of your fast EnvPool loop, you save the trained PyTorch `agent` state via `torch.save(agent.state_dict(), "ppo_envpool_pong.pth")`.
2. **Standard Gym Evaluation:** You write a separate script that instantiates a standard, slow Python `gymnasium` environment wrapped with `RecordVideo`. You load the saved `.pth` weights into an identical PyTorch network, and then step through the `gymnasium` environment using the frozen agent.

Because evaluation only takes a few minutes (compared to hours of training), the performance overhead of Python rendering during this assessment phase doesn't matter.

Below is an example of what that evaluation script would look like.
