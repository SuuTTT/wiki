# Launch & Debugging Guide

If you experience execution errors, hanging processes, or file path issues, use this standard operating procedure.

## The Issue: `[Errno 2] No such file or directory`

The error observed in `/workspace/logs/world_models_train.log`:
```text
python: can't open file '/workspace/01_world_models_vae_rnn.py': [Errno 2] No such file
```
**Cause:** The terminal was physically located at the root `/workspace` directory, but the launch command `python 01_world_models_vae_rnn.py` assumed we were already inside the `wiki/research-world-models/` folder. Consequently, `nohup` failed to find the script.

## The Solution: Standardizing the Launch Command

Always use absolute paths, or explicitly `cd` into the target directory before writing to the designated `logs/` folder. 

Run the following block exactly to safely kill any ghost processes, initialize the correct directory, and launch unbuffered training.

```bash
# 1. Kill any existing broken/hanging scripts
pkill -f 01_world_models_vae_rnn.py
pkill -f tensorboard

# 2. Enter the correct directory explicitly
cd /workspace/wiki/research-world-models

# 3. Create the log directory inside research-world-models
mkdir -p logs

# 4. Activate environment and launch training (unbuffered -u to force logs to flush)
source /venv/main/bin/activate
nohup python -u 01_world_models_vae_rnn.py > logs/world_models_train.log 2>&1 &

# 5. Launch Tensorboard
nohup tensorboard --logdir benchmark --host 0.0.0.0 --port 6006 > logs/tensorboard.log 2>&1 &

# 6. Monitor in real-time
tail -f logs/world_models_train.log
```

## Monitoring Checks
If the `tail -f` command shows the `tqdm` progress bar updating:
```text
Training updates:  0%|          | 0/488 [00:00<?, ?it/s]
```
The script is successfully running. Metrics will begin populating in TensorBoard on `http://localhost:6006` after the first update loop completes!