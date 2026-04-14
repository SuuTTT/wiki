import glob
import os

# Files that need basic video toggling and SummaryWriter to RLTracker swap
for file in glob.glob("/workspace/wiki/learn-cleanrl/*tutorial.py"):
    with open(file, "r") as f:
        content = f.read()
    
    # Disable video capture globally
    content = content.replace("capture_video: bool = True", "capture_video: bool = False")
    
    # Standardize writer if it exists
    if "from torch.utils.tensorboard import SummaryWriter" in content:
        content = content.replace(
            "from torch.utils.tensorboard import SummaryWriter", 
            "from cleanrl_utils.logger import RLTracker\nfrom torch.utils.tensorboard import SummaryWriter"
        )
        content = content.replace(
            'writer = SummaryWriter(f"runs/{run_name}")', 
            'tracker = RLTracker(args.exp_name, args.seed)\n    writer = tracker.writer'
        )
        # End hook
        if "writer.close()" in content and "tracker.save_checkpoint" not in content:
            content = content.replace(
                'writer.close()',
                'try:\n        tracker.save_checkpoint(agent.state_dict() if "agent" in locals() else q_network.state_dict())\n    except:\n        pass\n    writer.close()'
            )
            
    with open(file, "w") as f:
        f.write(content)

print("Standard tutorials patched.")
