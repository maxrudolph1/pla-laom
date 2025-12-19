#!/bin/bash

# run pla with clean backgrounds
python train_pla.py \
    --pla.custom_dataset=True \
    --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
    --name=12182025-test \
    --pla.discriminator_weight=0.01 \
    --seed=0 \
    --group=12182025-test \
    --pla.num_epochs=300 \
    --pla.la_regularization=softmax
