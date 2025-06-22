import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from mamba_models import Mamba, ModelArgs
from lstm_models import MTLSTM

import argparse
import time
import os
import sys
import numpy as np

from transformer_modules import MTTokenizer, MTStatePredictor, generate_causal_mask, MetadataEncoder
from MTDataset import MTDataset

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=3435, help="Seed for random numbers - good for reproducibility.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument("--embed_dim", type=int, default=8, help="Embedding dimensions for each CHARACTER in MT representation. 1 character = 1 dimer.")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads for model")
parser.add_argument("--vocab_size", type=int, default=4, help="Number of unique possible characters in a state")
parser.add_argument("--state_length", type=int, default=260, help="Number of characters in a state. 260 = 20*13 because 1 ring is 13 dimers, so MT length of 20 = 260 characters")
parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder blocks")
parser.add_argument("--hidden_dim", type=int, default=16, help="Size of hidden dimensions in encoder blocks")
parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
parser.add_argument("--save_path", type=str, required=True, help="Location to save models and outputs to.")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--model_type", type=str, default="transformer", choices=['transformer', 'mamba', 'lstm'], help="mamba model, lstm, or transformer. Pick one")


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer():
    def __init__(
        self,
        train_data: DataLoader=None,
        val_data: DataLoader=None,
        tokenizer: MTTokenizer=None,
        model: torch.nn.Module=None,
        criterion: nn.CrossEntropyLoss()=None,
        optimizer: optim.Optimizer=None,
        scheduler: optim.lr_scheduler=None,
        num_epochs: int=None,
        save_path: str=None,
        vocab_size: int=None,
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer # might not need this, but good to have
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.vocab_size = vocab_size
        self.epochs_run = 0
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.model = model.to(self.device)
        if os.path.exists(f"saved_models/{self.save_path}"):
            print(f"Rank {self.gpu_id} loading snapshot")
            self._load_snapshot(f"saved_models/{self.save_path}.pt")
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model = snapshot["MODEL"].to(self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        
        
    def _run_batch(self, batch):
        meta_data, seqs = batch
        meta_data = torch.tensor(meta_data, dtype=torch.float32).to(self.device)

        # Encode sequence with tokenizer
        seqs = self.tokenizer.encode(seqs).to(self.device).unsqueeze(0)

        targets = seqs
        inputs = seqs[:, :-1, :, :]
        
        outputs = self.model(inputs, meta_data)
        reshaped_outputs = outputs.view(-1, self.vocab_size)
        reshaped_targets = targets.view(-1)
        loss = self.criterion(reshaped_outputs, reshaped_targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        
    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        for batch in tqdm(self.train_data, desc="Training", total=len(self.train_data)):
            self._run_batch(batch)
        
        
    def train(self, args):
        # Initial eval to get baselines
        best_ce_loss = self.evaluate()
        
        for epoch in range(self.num_epochs):
            self.model.train() # set to train mode
            self._run_epoch(epoch)
            
            # Eval after each epoch
            avg_ce_loss = self.evaluate()
            current_lr = self.scheduler.get_last_lr()[0]
            
            if self.gpu_id == 0:
                self._log_progress(epoch=(epoch+1), avg_val_ce=avg_ce_loss, lr=current_lr, file_name=self.save_path)
                if avg_ce_loss < best_ce_loss:
                    best_ce_loss = avg_ce_loss
                    self._save_snapshot(epoch, args)
                
        
    def evaluate(self):
        self.model.eval()
        
        total_val_ce_loss = torch.tensor(0.0, device=self.device)
        total_preds = 0
        with torch.no_grad():
            for batch in tqdm(self.val_data, desc="Validation", total=len(self.val_data)):
                meta_data, seqs = batch

                meta_data = torch.tensor(meta_data, dtype=torch.float32).to(self.device)
                
                # Encode sequence with tokenizer
                seqs = self.tokenizer.encode(seqs).to(self.device).unsqueeze(0)
                
                targets = seqs
                inputs = seqs[:, :-1, :, :]
                preds_in_batch = seqs.size(0)*seqs.size(1)*seqs.size(2)
                
                outputs = self.model(inputs, meta_data)

                reshaped_outputs = outputs.view(-1, self.vocab_size)
                reshaped_targets = targets.view(-1)
                
                loss = self.criterion(reshaped_outputs, reshaped_targets)
                total_val_ce_loss += loss
                total_preds += preds_in_batch
        
        avg_ce_loss = total_val_ce_loss.item() / total_preds
        return avg_ce_loss
        
    def _save_snapshot(self, epoch, args):
        snapshot = {
            "ARGS": args.__dict__,
            "MODEL": self.model.module.to('cpu'),
            "EPOCHS_RUN": epoch,
        }
        directory = "saved_models/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        save_file = directory + self.save_path + ".pt"
        torch.save(snapshot, save_file)
        print(f"Epoch {epoch} | Training snapshot saved at {save_file}")
        
        
    def _log_progress(self, epoch=None, avg_val_ce=None, lr=None, file_name=None):
        # Ensure the directory exists
        directory = "progress_outputs/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{file_name}_progress_log.txt")

        with open(file_path, "a") as f:
            progress = f"Epoch: {epoch}, Val. Avg. CE Loss: {avg_val_ce}, Learning rate: {lr}\n"
            f.write(progress)


def single_collate(batch):
    # batch is a list of one sample:  [(meta, seqs)]
    return batch[0]
            
def prepare_dataloader(dataset: Dataset, batch_size: int=1, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=single_collate,
        num_workers=1,
        sampler=DistributedSampler(dataset)
    )
            

def load_train_objects(args):
    train_data = MTDataset(data_path=args.train_data)
    val_data = MTDataset(data_path=args.val_data)
    train_loader = prepare_dataloader(train_data)
    val_loader = prepare_dataloader(val_data)
    
    tokenizer = MTTokenizer()
    
    if args.model_type == "transformer":
        print("Training a transformer based model!")
        model = MTStatePredictor(
            vocab_size=args.vocab_size,
            state_length=args.state_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim
        )
    elif args.model_type == "mamba":
        print("Training a Mamba based model!")
        d_model = 260*args.embed_dim
        n_layer = args.num_layers
        vocab_size = len(tokenizer.vocab)
        print(f"Vocab size: {vocab_size}")
        model_args = ModelArgs(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size)
        model = Mamba(
            model_args,
            embed_dim=args.embed_dim)
        print(model_args)
    elif args.model_type == "lstm":
        print("Training an LSTM based model!")
        model = MTLSTM(
            hidden_dim=3072,
        )
        
    
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
          
    warmup_steps = 0.03*len(train_loader)
    def lr_lambda(current_step, warmup_steps=warmup_steps, total_steps=len(train_loader)*args.num_epochs):
        if current_step < warmup_steps:
            return current_step / warmup_steps  # Linear warmup
        return 0.5 * (1 + torch.cos(torch.tensor((current_step - warmup_steps) / (total_steps - warmup_steps) * 3.1416)))
                         
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return train_loader, val_loader, tokenizer, model, criterion, optimizer, scheduler



def main(args):
    ddp_setup()
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    train_data, val_data, tokenizer, model, criterion, optimizer, scheduler = load_train_objects(args)
    
    trainer = Trainer(
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        vocab_size=args.vocab_size,
    )
    trainer.train(args)
    
    destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    main(args)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total training time (min:sec): {minutes}:{seconds}")
