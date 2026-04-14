import os
import torch
import torch.nn.functional as F
import gymnasium as gym
import imageio
import numpy as np
from pathlib import Path
import sys

# Append the current directory so we can import the classes directly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from importlib import import_module
wm = import_module("01_world_models_vae_rnn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_obs(o):
    # From gym: (96, 96, 3) uint8 to PyTorch format (1, 3, 64, 64)
    o = torch.tensor(o, dtype=torch.float32).to(device) / 255.0
    o = o.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 96, 96)
    o = F.interpolate(o, size=(64, 64))  # (1, 3, 64, 64)
    return o

def generate_video(model_dir, output_file="world_model_agent.mp4"):
    print(f"Loading models from {model_dir}...")
    checkpoint_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Initialize components
    vae = wm.VAE(latent_dim=32).to(device)
    rnn = wm.MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, num_mix=5).to(device)
    agent = wm.ActorCritic(latent_dim=32, hidden_dim=256, action_dim=3).to(device)
    
    vae.load_state_dict(checkpoint['vae'])
    rnn.load_state_dict(checkpoint['rnn'])
    agent.load_state_dict(checkpoint['agent'])
    
    vae.eval()
    rnn.eval()
    agent.eval()

    print("Initializing environment...")
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, info = env.reset()
    
    hidden = (torch.zeros(1, 1, 256).to(device),
              torch.zeros(1, 1, 256).to(device))
              
    frames = []
    total_reward = 0
    done = False
    
    print("Simulating agent...")
    for t in range(500): # Hard cap at 500 frames for rendering speed
        frames.append(env.render())
        
        proc_obs = preprocess_obs(obs)
        with torch.no_grad():
            _, mu, logvar = vae(proc_obs)
            z = vae.reparameterize(mu, logvar)
            
            action, _, _, _ = agent.get_action_and_value(z, hidden[0].squeeze(0))
            _, _, _, next_hidden = rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
            hidden = next_hidden
            
        action_np = action.squeeze(0).cpu().numpy()
        # Bound actions physically (steering -1 to 1, gas 0 to 1, brake 0 to 1) 
        action_np[0] = np.clip(action_np[0], -1.0, 1.0)
        action_np[1] = np.clip(action_np[1], 0.0, 1.0)
        action_np[2] = np.clip(action_np[2], 0.0, 1.0)
        
        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += reward
        
        if terminated or truncated:
            break
            
    env.close()
    print(f"Episode finished. Total Reward: {total_reward:.2f}")
    
    print(f"Encoding video to {output_file}...")
    imageio.mimsave(output_file, frames, fps=30)
    print("Done!")

if __name__ == "__main__":
    import glob
    # Find latest benchmark folder
    search_path = os.path.join(os.path.dirname(__file__), "benchmark/01_world_models_vae_rnn/*")
    folders = sorted(glob.glob(search_path), key=os.path.getmtime, reverse=True)
    if folders:
        latest_folder = folders[0]
        out_video = os.path.join(os.path.dirname(__file__), "output_video.mp4")
        generate_video(latest_folder, out_video)
    else:
        print("No benchmarks found to visualize.")
