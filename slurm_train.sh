#!/bin/bash
#SBATCH --partition=gpu_p
#SBATCH --job-name=train_run_x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-
#SBATCH --mem=64G
#SBATCH --output=train_run_x.txt

#SBATCH --mail-type=END,BEGIN,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@uga.edu  # Where to send mail (change username@uga.edu to your email address)

ml purge
ml Python/3.11.3-GCCcore-12.3.0

source mtenv/bin/activate

# Potentially switch into directories if this repository is not in your home directory:
# cd to/my/mt/directory

torchrun --standalone --nproc_per_node=gpu train_model.py --train_data processed_split_sequences/train_sequences_1.pkl --val_data processed_split_sequences/test_sequences_1.pkl --save_path however_you_want_to_name_your_model --num_heads 8 --num_layers 1 --hidden_dim 256 --lr 5e-6 --model_type mamba --embed_dim 8

# Not all of the args apply to all models.
# Read args in train_model.py and see which ones are used for which model architectures near the bottom of that file.