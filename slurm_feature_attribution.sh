#!/bin/bash
#SBATCH --partition=gpu_p
#SBATCH --job-name=feature_attribution_x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-
#SBATCH --mem=64G
#SBATCH --output=feature_attribution_x.txt

#SBATCH --mail-type=END,BEGIN,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@uga.edu  # Where to send mail (change username@uga.edu to your email address)

ml purge
ml Python/3.11.3-GCCcore-12.3.0

source mtenv/bin/activate

# Potentially switch into directories if this repository is not in your home directory:
# cd to/my/mt/directory

python feature_attribution.py --model mamba --model_name saved_models/whatever_your_model_is_named.pt