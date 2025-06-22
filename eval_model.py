import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer_modules import MTTokenizer, MTStatePredictor, generate_causal_mask
from MTDataset import MTDataset
import argparse
import sys
from tqdm import tqdm
import thop
from thop import profile, clever_format

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=3435, help="Seed for random numbers - good for reproducibility.")
parser.add_argument("--model", type=str, default="saved_models/model_small_lr_pos_encoding_8heads_24layers_hd64.pt", help="Path to the model you want to evaluate.")
parser.add_argument("--val_data", type=str, default="processed_split_sequences/test_sequences_1.pkl", help="Path to validation dataset.")
parser.add_argument("--eval_type", type=str, default="total", choices=["total", "changes"], help="Total for overall accuracy, changes, for accuracy on positions that changed")

if torch.cuda.is_available():
    device = "cuda"
    print("Cuda is available. Using GPU.")
else:
    device = "cpu"
    print("Cuda is not available. Using CPU.")


def load_model(path):
    """
    This function loads the model.
    
    Args:
        path (str): Path to the saved model you wish to evaluate.
        
    Returns:
        nn.Module: The saved nn.Module that is your model. 
                   May have saved it as a dictionary with the key 'MODEL'.
    """
    model = torch.load(path, weights_only=False)
    if isinstance(model, dict):
        model = model['MODEL'].to(device)
        return model
    return model.to(device)


def load_data(path):
    """
    This function loads the data you want to use to evaluate your model on.
    
    Args:
        path (str): Path to the dataset
        
    Returns:
        DataLoader: A PyTorch DataLoader object that allows you to 
                    iterate over the sequences in the dataset.
    """
    val_data = MTDataset(data_path=path)
    val_dl = prepare_dataloader(val_data)
    return val_dl
    
    
def prepare_dataloader(dataset: Dataset, batch_size: int=1, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=shuffle,
        collate_fn=single_collate,
        num_workers=1
    )

def single_collate(batch):
    # batch is a list of one sample:  [(meta, seqs)]
    return batch[0]
    

def evaluate(model: nn.Module, data: DataLoader):
    tokenizer = MTTokenizer()
    vocab_size = len(tokenizer.vocab)
    model.eval()
    
    num_preds = 0
    num_correct = 0
    step_size = 500
    with torch.no_grad():
        for batch in tqdm(data, desc="Validation", total=len(data)):
            meta_data, seqs = batch
            
            meta_data = torch.tensor(meta_data, dtype=torch.float32).to(device)
            seqs = tokenizer.encode(seqs).to(device).unsqueeze(0)
         
            targets = seqs.reshape(-1)
            
            inputs = seqs[:, :-1, :, :]

            preds_in_batch = seqs.size(0)*seqs.size(1)*seqs.size(2)
            
            outputs = model(inputs, meta_data)
            
            reshaped_outputs = outputs.view(-1, vocab_size)
            _, predicted_ids = torch.max(reshaped_outputs, dim=-1)
            
            for start in range(0, predicted_ids.size(0), step_size):
                # Check accuracy in chunks to save memory
                end = start + step_size
                sub_preds = predicted_ids[start:end]
                sub_targets = targets[start:end]
                num_preds += len(sub_preds)
                
                # This builds a massive boolean mask, which takes up a lot of memory, hence why doing this in chunks
                num_correct += (sub_preds == sub_targets).sum()
            
    accuracy = (100 * (num_correct/num_preds)).item()
    return accuracy


def eval_changed_dimers(model, data):
    tokenizer = MTTokenizer()
    vocab_size = len(tokenizer.vocab)
    model.eval()
    
    total_correct = 0
    total_changes = 0
    with torch.no_grad():
        for batch in tqdm(data, desc="Validation", total=len(data)):
            meta_data, seqs = batch
            
            meta_data = torch.tensor(meta_data, dtype=torch.float32).to(device)
            seqs = tokenizer.encode(seqs).to(device).unsqueeze(0)
         
            targets = seqs.reshape(-1)
            
            inputs = seqs[:, :-1, :, :]
            preds_in_batch = seqs.size(0)*seqs.size(1)*seqs.size(2)
            
            outputs = model(inputs, meta_data)
            
            reshaped_outputs = outputs.view(-1, vocab_size)
            _, predicted_ids = torch.max(reshaped_outputs, dim=-1)
            
            predicted_ids = predicted_ids.view(-1, 260)
            inputs = inputs.squeeze(0).squeeze(-1)
            
            diffs = torch.zeros_like(inputs)
            diffs[1:] = inputs[1:] - inputs[:-1]
            
            mask = diffs != 0
            indices = torch.nonzero(mask, as_tuple=False)
            num_changes = indices.size(0)
            
            rows = indices[:, 0]
            cols = indices[:, 1]
            
            pred_vals = predicted_ids[rows, cols]
            actual_vals = inputs[rows, cols]
            
            correct_mask = pred_vals == actual_vals
            num_correct = correct_mask.sum().item()
            total_correct += num_correct
            total_changes += num_changes
    
    print(f"Total number of correct: {total_correct}")
    print(f"Total number of changes: {total_changes}")
    accuracy = (total_correct / total_changes)*100
    return accuracy
            

def main(args):
    print("Loading model!")
    model = load_model(args.model)
    print("Loading data!")
    data = load_data(args.val_data)
    
    # First, profile the model for size
    rand_sequence = torch.randint(4, (1, 2378, 260, 1)).to(device)
    rand_metadata = torch.rand(23).to(device)
    
    # Profile the model to count MACs
    print("Profiling model first!")
    macs, params = profile(model, inputs=(rand_sequence,rand_metadata))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}\n\n")
    
    
    print("Running eval!")
    if args.eval_type == "total":
        accuracy = evaluate(model, data)
        print(f"Accuracy on validation data: {accuracy}")
    elif args.eval_type == "changes":
        per_change_acc = eval_changed_dimers(model, data)
        print(f"Accuracy on dimers that actually changed: {per_change_acc}")

    
if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)