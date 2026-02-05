#!/bin/bash

# run pla with clean backgrounds
for i in 0.0 0.0001 0.001 0.01 0.1 1.0 10.0; do
    echo "Running with behavior loss coef $i"
    python train_pla.py \
    --pla.custom_dataset=True \
    --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
    --name=01152026-pla \
    --seed=0 \
    --group=01152026-pla-fine-tuned \
    --pla.num_epochs=300 \
    --fine_tune=True \
    --ckpt_path=artifacts/distracting-backgrounds/base.pth \
    --pla.alignment_method=discriminator \
    --pla.behavior_loss_coef=$i --slurm
done
