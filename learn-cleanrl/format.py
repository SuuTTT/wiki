import glob
for file in glob.glob("/workspace/wiki/learn-cleanrl/*tutorial.py"):
    with open(file, "r") as f:
        text = f.read()
    
    if "if \"v_loss\" in locals() and \"pg_loss\" in locals():" in text:
        # PPO tutorials end with a 4-space indent for the loop (for global_step...)
        # We need an 8-space indent for tracker.log_metrics under the `if` block, which is at 8-spaces. Wait, if the `if` block is at 8-spaces, then its contents must be at 12 spaces.
        old = """        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {"""
            
        new = """        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {"""
            
        # Let's cleanly replace the entire end block
        import re
        text = re.sub(
            r'[\ \t]*if "v_loss" in locals\(\) and "pg_loss" in locals\(\):[\s\S]*?envs\.close\(\)',
            '''        if "v_loss" in locals() and "pg_loss" in locals():
            tracker.log_metrics("losses", {
                "value_loss": v_loss.item(), 
                "policy_loss": pg_loss.item(), 
                "entropy": entropy_loss.item()
            })
            tracker.log_sps()

    envs.close()''',
            text
        )
        
        with open(file, "w") as f:
            f.write(text)
print("Indentation fixed.")
