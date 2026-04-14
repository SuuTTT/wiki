import os
import time
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

class RLTracker:
    def __init__(self, exp_name, seed):
        timestamp = int(time.time())
        date_prefix = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        new_name = f"{date_prefix}-{timestamp}"
        
        self.log_dir = os.path.join("benchmark", exp_name, new_name)
        
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.start_time = time.time()
        print(f"Logging to {self.log_dir}")

    def step(self, num_steps=1):
        self.global_step += num_steps

    def log_episode(self, return_val, length=0, win=None):
        self.writer.add_scalar("charts/episodic_return", return_val, self.global_step)
        if length > 0:
            self.writer.add_scalar("charts/episodic_length", length, self.global_step)
        if win is not None:
            self.writer.add_scalar("charts/win_rate", win, self.global_step)

    def log_metrics(self, category, metrics_dict):
        """category can be 'losses', 'charts', etc."""
        for key, value in metrics_dict.items():
            self.writer.add_scalar(f"{category}/{key}", value, self.global_step)
            
    def log_sps(self):
        sps = int(self.global_step / max(time.time() - self.start_time, 1e-3))
        self.writer.add_scalar("charts/SPS", sps, self.global_step)
        return sps

    def save_checkpoint(self, state_dict, path=""):
        if not path:
            path = os.path.join(self.log_dir, "model.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        print(f"Model saved to {path} for video rendering.")

    def close(self):
        self.writer.close()
