#!/bin/bash

# run pla with distracting backgrounds
for i in {1..2}; do
for weight in {0.0,0.001,0.01}; do
for regularization in {l2,l1,softmax,none}; do
    echo "Running with discriminator weight $weight"
    echo "Running with seed $i"
    echo "Running with regularization $regularization"
    python train_pla.py \
        --pla.custom_dataset=True \
        --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
        --name=12232025-state-regularization-sweep \
        --pla.discriminator_weight=$weight \
        --seed=$i \
        --group=12232025-state-regularization-sweep \
        --pla.num_epochs=300 \
        --pla.state_regularization=$regularization \
        --slurm
done
done
done