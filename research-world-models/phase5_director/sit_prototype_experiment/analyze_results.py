import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_final_reward(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    found_tag = None
    all_tags = event_acc.Tags()['scalars']
    for tag in all_tags:
        if 'charts/episodic_return' in tag:
            found_tag = tag
            break
    if not found_tag:
        return None
    events = event_acc.Scalars(found_tag)
    return events[-1].value, events[-1].step

log_root = "/workspace/logs/robust_study"
results = []

for folder in sorted(os.listdir(log_root)):
    path = os.path.join(log_root, folder)
    if os.path.isdir(path):
        val = get_final_reward(path)
        if val:
            rew, step = val
            # robust_4x4_flat_fTrue_s11_1776678889
            # robust_8x8_sparse_flat_fTrue_s11_1776678908
            parts = folder.split('_')
            
            # Robust split logic
            if 'sparse' in parts:
                map_v = "8x8_sparse"
                mode_v = "flat" if "flat" in parts else "hierarchical"
                seed_v = [p for p in parts if p.startswith('s') and len(p) > 1][0]
            elif '8x8' in parts:
                map_v = "8x8"
                mode_v = "flat" if "flat" in parts else "hierarchical"
                seed_v = [p for p in parts if p.startswith('s') and len(p) > 1][0]
            elif '4x4' in parts:
                map_v = "4x4"
                mode_v = "flat" if "flat" in parts else "hierarchical"
                seed_v = [p for p in parts if p.startswith('s') and len(p) > 1][0]
            else:
                continue

            results.append({
                'Map': map_v,
                'Mode': mode_v,
                'Seed': seed_v,
                'Final_Reward': rew,
                'Steps': step
            })

if not results:
    print("No results found.")
    exit()

df = pd.DataFrame(results)
print("### Debug DataFrame ###")
print(df)

summary = df.groupby(['Map', 'Mode'])['Final_Reward'].agg(['mean', 'std']).reset_index()
print("\n### Robustness Study Summary ###")
print(summary.to_markdown())
print("\n### Complete Results ###")
print(df.to_markdown())

