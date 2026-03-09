#!/bin/bash

# Parameters
#SBATCH --account=amyzhang
#SBATCH --cpus-per-task=16
#SBATCH --error=slurm_jobs/job_%j/err.err
#SBATCH --output=slurm_jobs/job_%j/out.out
#SBATCH --job-name=pla-pretrain
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --partition=allnodes
#SBATCH --time=6:00:00

source /u/mrudolph/miniconda3/etc/profile.d/conda.sh
conda activate pla-laom
python train_pla.py --pla.custom_dataset=True --pla.data_path=data/pm_medium_distracting/policy_rollouts/15_policies/12_backgrounds/action_repeat_1/8192_episodes/8192_episodes.hdf5 --name=pla-02052026-pretrained --seed=0 --group=pla-02052026-pretrained --pla.num_epochs=300 --fine_tune=False --pretrain=True
