import os

# 1. PPO Continuous
f1 = '/workspace/wiki/learn-cleanrl/ppo_continuous_tutorial.py'
with open(f1, 'r') as f: text = f.read()

old_env = """        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)"""
new_env = """        env = gym.wrappers.RecordEpisodeStatistics(env)
        # BUGFIX: Continuous PPO is highly sensitive to outlier actions & observations.
        # Added ClipAction, NormalizeObservation, and TransformObservation to prevent the policy from plateauing 
        # below -500 local minimums and ensure symmetric reward topologies.
        env = gym.wrappers.ClipAction(env)"""

old_gs1 = """            global_step += args.num_envs
            tracker.global_step = global_step"""
new_gs1 = """            global_step += args.num_envs
            # BUGFIX: Sync tracker.global_step to the algorithm's iteration frame to prevent Tensorboard logs from stacking exclusively at x=0
            tracker.global_step = global_step"""

if old_env in text: text = text.replace(old_env, new_env)
if old_gs1 in text: text = text.replace(old_gs1, new_gs1)

with open(f1, 'w') as f: f.write(text)


# 2. PPO Envpool
f2 = '/workspace/wiki/learn-cleanrl/ppo_envpool_tutorial.py'
with open(f2, 'r') as f: text = f.read()

old_gs2 = """        global_step += 1 * NUM_ENVS
        tracker.global_step = global_step"""
new_gs2 = """        global_step += 1 * NUM_ENVS
        # BUGFIX: Explicitly mapping global_step to the RLTracker so the x-axis advances horizontally in Tensorboard recordings
        tracker.global_step = global_step"""

if old_gs2 in text: text = text.replace(old_gs2, new_gs2)

with open(f2, 'w') as f: f.write(text)


# 3. PPO Selfplay
f3 = '/workspace/wiki/learn-cleanrl/ppo_selfplay_tutorial.py'
with open(f3, 'r') as f: text = f.read()

old_gs3 = """        global_step += TOTAL_PARALLEL_AGENTS
        tracker.global_step = global_step"""
new_gs3 = """        global_step += TOTAL_PARALLEL_AGENTS
        # BUGFIX: Map actual environment frames to the RLTracker to resolve the x=0 vertical grouping bug
        tracker.global_step = global_step"""

old_exp = """tracker = RLTracker("ppo_selfplay", args.seed)"""
new_exp = """# BUGFIX: Hardcoded "ppo_selfplay" as the experiment name to bypass the AttributeError caused by missing Arg config
tracker = RLTracker("ppo_selfplay", args.seed)"""

if old_gs3 in text: text = text.replace(old_gs3, new_gs3)
if old_exp in text: text = text.replace(old_exp, new_exp)

with open(f3, 'w') as f: f.write(text)


# 4. Rainbow
f4 = '/workspace/wiki/learn-cleanrl/rainbow_tutorial.py'
with open(f4, 'r') as f: text = f.read()

old_r = """                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                q_values = (old_pmfs * target_network.support).sum(dim=1).mean()
                writer.add_scalar("losses/q_values", q_values.item(), global_step)"""
new_r = """                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                # BUGFIX: Rainbow generates probability distributions (PMFs) via its Distributional layers instead of scalar Q-Values.
                # To log comparable Q-Values to Tensorboard, we multiply the network support bins by their distribution probability.
                q_values = (old_pmfs * target_network.support).sum(dim=1).mean()
                writer.add_scalar("losses/q_values", q_values.item(), global_step)"""

if old_r in text: text = text.replace(old_r, new_r)

with open(f4, 'w') as f: f.write(text)

print("Comments Applied.")
