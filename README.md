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
### Either use the run_trainer.sh script directly from the command line, or (if working on an HPC with a slurm scheduler) use the slurm_train.sh script.
