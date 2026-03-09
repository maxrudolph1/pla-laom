#!/bin/bash

# Set GPU device (change to desired GPU index)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# run pla with clean backgrounds
python adapt_pla.py \
    --pla.custom_dataset=True \
    --pla.data_path=data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
    --ckpt_path=artifacts/pla-02052026-pretrained/pla-02052026-pretrained-7cc288fb-3e87-47da-8553-544f504df38c/pretrained.pth \
    --name=pla-02052026-pretrained-adapt \
    --seed=2 \
    --group=pla-02052026-pretrained-adapt \
    --pla.num_epochs=300 