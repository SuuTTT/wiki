#!/bin/bash
source /venv/main/bin/activate
cd /workspace/wiki/learn-cleanrl

# Create a logs directory
mkdir -p logs

echo "Starting performance runs with nohup..."

TUTORIALS=(
    "rainbow_tutorial.py"
    "ppo_lstm_tutorial.py"
    "ppo_continuous_tutorial.py"
    "ppg_tutorial.py"
    "ppo_selfplay_tutorial.py"
    "ppo_envpool_tutorial.py"
)

for file in "${TUTORIALS[@]}"; do
    if [ -f "$file" ]; then
        echo "Launching $file in background..."
        # We run via nohup
        nohup python "$file" > "logs/${file}.log" 2>&1 &
        echo "PID $! -> logs/${file}.log"
    else
        echo "File $file not found!"
    fi
done

echo "All tutorials launched! Check the logs/ directory for outputs."
