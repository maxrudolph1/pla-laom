#!/bin/bash

# run pla with clean backgrounds
python train_pla.py \
    --pla.custom_dataset=True \
    --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/50_episodes/50_episodes.hdf5 \
    --name=test