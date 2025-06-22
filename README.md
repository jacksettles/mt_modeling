# Microtubule modeling with deep sequence models
My attempt at modeling microtubule dynamic instability using an LSTM, a transformer, and a mamba SSM.



## 1. Create virtual environment (use any name you like)
python -m venv mtenv

## 2. Activate the virtual environment
### On macOS/Linux:
source mtenv/bin/activate

### On Windows:
mtenv\Scripts\activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Run training scripts:
Either use the run_trainer.sh script directly from the command line, or (if working on an HPC with a slurm scheduler) use the slurm_train.sh script.
This allows the training job to run on its own node so you don't have to wait on it to finish training.

<pre>
  ```bash
  sbatch slurm_train.sh
  ```
</pre>

Training file has been set up to run in a PyTorch DDP fashion should you have access to and decide to use more than 1 GPU.
If using more than one GPU, adjust the slurm_train.sh script to allocate more than 1 device. For example, to use 2 A100 GPUs:

<pre>
#!/bin/bash
#SBATCH --partition=gpu_p
#SBATCH --job-name=train_run_x
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --time=1-
#SBATCH --mem=64G
#SBATCH --output=train_run_x.txt

#SBATCH --mail-type=END,BEGIN,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@uga.edu

</pre>




