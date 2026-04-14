import glob

for file in glob.glob("/workspace/wiki/learn-cleanrl/*tutorial.py"):
    if "envpool" in file or "selfplay" in file or "rainbow" in file or "c51" in file or "dqn" in file or "ddpg" in file or "sac" in file:
        continue
    
    with open(file, "r") as f:
        text = f.read()
    
    if "tracker.log_sps()" not in text:
        text = text.replace(
            "envs.close()",
            """        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {
                "value_loss": v_loss.item(), 
                "policy_loss": pg_loss.item(), 
                "entropy": entropy_loss.item()
            })
            tracker.log_sps()
            
    envs.close()"""
        )
        
        with open(file, "w") as f:
            f.write(text)

print("Losses patched into PPO models")
