#!/bin/bash
#SBATCH --job-name=pla
#SBATCH --output=slurm_scripts/job_%j.out
#SBATCH --error=slurm_scripts/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=amyzhang
#SBATCH --partition=mll

source /scratch/cluster/mrudolph/miniconda3/etc/profile.d/conda.sh
conda activate pla-laom

# run pla with distracting backgrounds
python train_pla.py \
    --pla.custom_dataset=True \
    --pla.data_path=/u/mrudolph/documents/pla-laom/data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 \
    --name=pla-distracting-medium-discriminator \
    --pla.discriminator_weight=0.01 \
    --seed=0