import glob

# For all tutorials using RLTracker, we need to inject:
# tracker.global_step = global_step
# right before tracker.log_metrics or tracker.log_episode or tracker.log_sps

scripts = glob.glob("/workspace/wiki/learn-cleanrl/*_tutorial.py")
for script in scripts:
    with open(script, "r") as f:
        text = f.read()
    
    # Simple replace for log_sps since it's common in PPO loops
    # Or replace wherever global_step is incremented, but it changes by algorithm.
    # The safest way is just `tracker.global_step = global_step` right before `tracker.log_` calls
    # or just before tracker.log_sps()
    
    text = text.replace('tracker.log_sps()', 'tracker.global_step = global_step\n        tracker.log_sps()')
    
    with open(script, "w") as f:
        f.write(text)
        
