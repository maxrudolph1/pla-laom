#!/bin/bash
# run pla with distracting backgrounds
# for i in {1..2}; do
#     python train_pla.py \
#         --pla.custom_dataset=True \
#         --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
#         --name=base-pla-model \
#         --pla.discriminator_weight=0.0 \
#         --seed=$i \
#         --group=distracting-backgrounds \
#         --pla.num_epochs=300 \
#         --slurm
# done

for i in {1..2}; do
    python train_pla.py \
        --pla.custom_dataset=True \
        --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_size/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
        --name=base-pla-model \
        --pla.discriminator_weight=0.0 \
        --seed=$i \
        --group=clean-backgrounds \
        --pla.num_epochs=300 \
        --slurm
done