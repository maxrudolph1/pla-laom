#!/bin/bash

# run pla with distracting backgrounds
for i in {1..5}; do
for weight in {0.001,0.01}; do
for regularization in {l2,l1,softmax,none}; do
    echo "Running with weight $weight"
    echo "Running with seed $i"
    echo "Running with regularization $regularization"
    python train_pla.py \
        --pla.custom_dataset=True \
        --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
        --name=12182025-regularization-sweep \
        --pla.discriminator_weight=$weight \
        --seed=$i \
        --group=12182025-regularization-sweep \
        --pla.num_epochs=300 \
        --pla.la_regularization=$regularization \
        --slurm
done
done
done