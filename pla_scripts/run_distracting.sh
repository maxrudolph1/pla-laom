#!/bin/bash

# run pla with distracting backgrounds

for weight in {0.0,0.001,0.01,0.1}; do
    echo "Running with weight $weight"
for i in {1..5}; do
    echo "Running with seed $i"
python train_pla.py \
    --pla.custom_dataset=True \
    --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
    --name=discriminator-sweep \
    --pla.discriminator_weight=$weight \
    --seed=$i \
    --slurm
done
done