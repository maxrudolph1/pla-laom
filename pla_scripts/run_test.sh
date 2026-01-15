#!/bin/bash

# run pla with clean backgrounds
python train_pla.py \
    --pla.custom_dataset=True \
    --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
    --name=01152026-pla \
    --seed=0 \
    --group=01152026-pla-pre-trained \
    --pla.num_epochs=300 \
    --fine_tune=False \
    --pretrain=True \
    # --ckpt_path=artifacts/distracting-backgrounds/base-pla-model-dfa24071-5702-487e-bd2a-e27dc58858d6/lapo.pth
