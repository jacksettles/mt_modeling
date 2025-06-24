# Microtubule modeling with deep sequence models
My attempt at modeling microtubule dynamic instability using an LSTM, a transformer, and a mamba SSM.



## 1. Create virtual environment (use any name you like, here we use 'mtenv')
python -m venv mtenv

## 2. Activate the virtual environment
### On macOS/Linux:
source mtenv/bin/activate

### On Windows:
mtenv\Scripts\activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Run training scripts:
Either use the run_trainer.sh script directly from the command line via `source run_trainer.sh`, or (if working on an HPC with a slurm scheduler) use the slurm_train.sh script via `sbatch slurm_train.sh`. This allows the training job to run on its own node so you don't have to wait on it to finish training.


Make sure to check the args in the command inside the slurm_train.sh (or run_trainer.sh) file so that you have are training with the configuration you want.
If using the slurm_train.sh script, this will launch a job on a new node, so more than likely it will open in your home directory. Make sure you add a 
`cd path/to/your/mt/directory` line before trying to run the training file.

Training file has been set up to run in a PyTorch DDP fashion should you have access to and decide to use more than 1 GPU.
If using more than one GPU, adjust the slurm_train.sh script to allocate more than 1 device. For example, to use 2 A100 GPUs:

<pre>
```
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

```
</pre>

## 5. Feature Attribution

If you want to use the `input*gradient` method of feature importance attribution, run the slurm_feature_attribution.sh script.
Again check the args (make sure you pass in the path to a model that actually exists).
Once completed, you should be able to load these saved feature attributions into the
`visualize_feature_importance.ipynb` notebook to run the animation.

## Credit where credit is due
The implementation for the Mamba models comes from this github repo: https://github.com/johnma2006/mamba-minimal/tree/master . I simply took the code
from this repo and modified it to fit with the use case of MT modeling instead of language modeling.

## 6. Deactivate virtual environment when done
`deactivate`


