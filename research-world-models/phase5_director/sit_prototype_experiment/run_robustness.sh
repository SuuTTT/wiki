#!/bin/bash
mkdir -p /workspace/logs/robust_study

declare -a MAPS=("4x4" "8x8" "8x8_sparse")
declare -a MODES=("flat" "hierarchical")
declare -a SEEDS=(11 22 33)

echo "Starting Robustness Study: ${#MAPS[@]} maps x ${#MODES[@]} modes x ${#SEEDS[@]} seeds"

for map in "${MAPS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Queuing: Map=$map Mode=$mode Seed=$seed"
      nohup python3 /workspace/wiki/research-world-models/phase5_director/sit_prototype_experiment/sit_robustness_study.py --map "$map" --mode "$mode" --seed "$seed" >> /workspace/wiki/research-world-models/phase5_director/sit_prototype_experiment/robustness_stdout.log 2>&1 &
      # Sleep briefly to avoid resource spikes during initialization
      sleep 1
    done
  done
done

echo "Experiments running in background. Track via TensorBoard at /workspace/logs/robust_study"
