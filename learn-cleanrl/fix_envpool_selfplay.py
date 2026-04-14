import re

# FIX SELFPLAY
with open("/workspace/wiki/learn-cleanrl/ppo_selfplay_tutorial.py", "r") as f:
    sp_text = f.read()

# Remove the broken imports
sp_text = re.sub(r'from cleanrl_utils\.logger import RLTracker\.nn as nn', 'import torch.nn as nn', sp_text)
sp_text = re.sub(r'from cleanrl_utils\.logger import RLTracker\.optim as optim', 'import torch.optim as optim', sp_text)

# We need to make sure `tracker = ` is actually present
if "tracker = RLTracker(" not in sp_text:
    sp_text = sp_text.replace('device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")',
                              'tracker = RLTracker(args.exp_name, args.seed)\n    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")')

with open("/workspace/wiki/learn-cleanrl/ppo_selfplay_tutorial.py", "w") as f:
    f.write(sp_text)

# FIX ENVPOOL
with open("/workspace/wiki/learn-cleanrl/ppo_envpool_tutorial.py", "r") as f:
    ep_text = f.read()

if "tracker = RLTracker(" not in ep_text:
    # Inject initialization near device
    ep_text = re.sub(r'(device = torch\.device\("cuda"\))',
                     r'tracker = RLTracker("ppo_envpool", 1)\n\1',
                     ep_text)

    # Replace prints
    ep_text = re.sub(r'print\(f"global_step=\{global_step\}, ep_return=\{info\[\'reward\'\]\[i\]\}"\)',
                     r'tracker.log_episode(info["reward"][i], 0, None)',
                     ep_text)
                     
    ep_text = re.sub(r'print\(f"Iteration \{update\}/\{num_updates\} - SPS: \{int\(global_step / \(time\.time\(\) - start_time\)\)\}"\)',
                     r'tracker.log_sps()\n        tracker.log_metrics("losses", {"value_loss": v_loss.item(), "policy_loss": pg_loss.item(), "entropy": entropy_loss.item()})',
                     ep_text)
                     
    ep_text = ep_text.replace('print("✅ EnvPool PPO Tutorial Completed!")', 
                              'tracker.save_checkpoint(agent.state_dict())\n    tracker.close()\n    print("✅ EnvPool PPO Tutorial Completed!")')

with open("/workspace/wiki/learn-cleanrl/ppo_envpool_tutorial.py", "w") as f:
    f.write(ep_text)
