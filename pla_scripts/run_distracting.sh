#!/bin/bash

# run pla with distracting backgrounds
python train_pla.py \
    --lapo.custom_dataset=True \
    --lapo.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
    --name=pla-distracting-medium-size