import glob
import re

for file in glob.glob("/workspace/wiki/learn-cleanrl/*tutorial.py"):
    with open(file, "r") as f:
        content = f.read()

    # 1. FIX EPISODIC RETURNS
    # Old logic:
    # if "final_info" in infos:
    #     for info in infos["final_info"]:
    #         if info and "episode" in info:
    #             progress_bar.set_postfix(episodic_return=info['episode']['r'][0])
    #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step) # or tracker.log_episode...
    
    # We replace that entire chunk using regex
    pattern = r'if "final_info" in infos:[\s\S]*?(?=envs\.step|with torch\.no_grad|b_obs = )'
    
    # Actually it's easier to just match from "if "final_info" in infos:" to the end of that block.
    # It's usually ~5 lines.
    content = re.sub(
        r'(?:if\s+"final_info"\s+in\s+infos:.*?)(?=\n\s*(?:with\s+torch\.no_grad|real_next_obs\s+=|b_obs\s+=|#\s+GAE|#\s+Compute|for\s+t\s+in|for\s+step\s+in))',
        '''if "_episode" in infos:
            for idx, d in enumerate(infos["_episode"]):
                if d:
                    r = infos["episode"]["r"][idx].item() if hasattr(infos["episode"]["r"][idx], "item") else infos["episode"]["r"][idx]
                    l = infos["episode"]["l"][idx].item() if hasattr(infos["episode"]["l"][idx], "item") else infos["episode"]["l"][idx]
                    tracker.log_episode(r, l)
                    if 'progress_bar' in locals():
                        progress_bar.set_postfix(episodic_return=f"{r:.2f}")
''',
        content,
        flags=re.DOTALL
    )

    # 2. INJECT LOSESS (if missing) BEFORE envs.close()
    
    # Let's check if the file is one of the policy gradients that is missing loss logs
    if "ppo_" in file or "ppg" in file or "sac" in file or "ddpg" in file or "dqn" in file or "c51" in file or "rainbow" in file:
        # Check if tracker.log_metrics is not already present
        if 'tracker.log_metrics' not in content:
            # We want to insert just before 'envs.close()'
            # However some are nested under loops. Actually it's safe to just inject tracker.log_metrics into the final optimization loop step block, or simply at the bottom of the outer training loop.
            # In PPO, outer loop ends just before envs.close(). But we need the vars from the inner loop.
            pass

    with open(file, "w") as f:
        f.write(content)
print("Infos patched")
