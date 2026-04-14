with open('/workspace/wiki/learn-cleanrl/ppo_selfplay_tutorial.py', 'r') as f:
    text = f.read()

import re
text = re.sub(r'tracker = RLTracker\(args.exp_name, args.seed\)\n\s+device = torch.device\("cuda"',
              'args = Args()\ntracker = RLTracker(args.exp_name, args.seed)\ndevice = torch.device("cuda"', text)

with open('/workspace/wiki/learn-cleanrl/ppo_selfplay_tutorial.py', 'w') as f:
    f.write(text)

with open('/workspace/wiki/learn-cleanrl/ppo_envpool_tutorial.py', 'r') as f:
    text = f.read()

text = re.sub(r'tracker = RLTracker\("ppo_envpool", 1\)\n\s+device',
              'tracker = RLTracker("ppo_envpool", 1)\ndevice', text)

with open('/workspace/wiki/learn-cleanrl/ppo_envpool_tutorial.py', 'w') as f:
    f.write(text)
