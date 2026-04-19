# Verification & Training Diagnostics: CarRacing-v3 PPO-LSTM

This document outlines the key metrics and failure indicators for the `ppo_lstm_zoo.py` baseline. Use this to verify the health of the 4M timestep training run.

## 1. Key Performance Indicators (KPIs)

| Metric | Target (4M Steps) | Description |
| :--- | :--- | :--- |
| `charts/episodic_return` | **> 850** | Final converged reward. |
| `charts/SPS` | **80 - 150** | Steps per second on GPU. |
| `losses/explained_variance` | **> 0.8** | Indicates the Critic accurately predicts returns. |
| `losses/approx_kl` | **< 0.02** | Stability of policy updates. |

## 2. Failure Indicators (Red Flags)

| Symptom | Indicator | Potential Cause |
| :--- | :--- | :--- |
| **Collapse** | Reward stays below 0 after 1M steps | Learning rate too high or ent_coef too low. |
| **Instability** | Sudden drop in reward with high KL | Policy update step too large; reduce `clip_coef`. |
| **Vanishing Gradients** | `explained_variance` stays near 0 | CNN/LSTM initialization issues or architecture mismatch. |
| **Staleness** | `SPS` significantly drops | Resource contention or memory leak in `SyncVectorEnv`. |

## 3. Metrics to Monitor in Tensorboard

### Early Stage (0 - 500k steps)
- **Reward**: Should rise from -80 to at least 0.
- **Entropy**: Should gradually decrease from its initial value.
- **Value Loss**: May spike initially but should stabilize.

### Late Stage (2M - 4M steps)
- **Explained Variance**: Should climb toward 1.0 as the environment becomes more predictable.
- **Episodic Length**: Should increase and then stabilize as the car avoids crashing/spinning out.

## 4. Post-Training Verification
Expected final reward distribution:
- **Mean**: 860+
- **Std Dev**: < 100
- **Success Rate**: Car should complete the lap without significant off-track excursions in 95% of evaluation episodes.

---
*Note: For further modification, if reward plateaus below 700, consider increasing `frame_stack` to 4 or adjusting `num_steps` to 256 to increase the LSTM's temporal horizon.*
