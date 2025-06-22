import torch
import torch.nn as nn
from mamba_models import Mamba
from transformer_modules import MTStatePredictor, MTTokenizer
from MTDataset import MTDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pickle
import argparse
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=3455, help="Seed for setting randomness")
parser.add_argument("--model", type=str, required=True, choices=['transformer', 'mamba'], help="Model architecture type")
parser.add_argument("--model_name", type=str, required=True, help="Name of the '.pt' file that the model is saved to within the 'saved_models/' subdirectory. Saved .pt file should be a dictionary with the model saved in a 'MODEL' key. Include the 'saved_models/' part of the file name for now.")
parser.add_argument("--data", type=str, default="processed_split_sequences/test_sequences_1.pkl", help="Path to data to perform LRP on.")

if torch.cuda.is_available():
    device = "cuda"
    print("Cuda is available. Using GPU.")
else:
    device = "cpu"
    print("Cuda not available. Using CPU.")

def load_objects(args):
    if args.model == "transformer":
        model_path = f"{args.model_name}"
    elif args.model == "mamba":
        model_path = f"{args.model_name}"
    else:
        print("Please specify in args either 'transformer' or 'mamba' for model.")
        sys.exit(0)
    print(f"Relevance prop on {args.model}")
    model = torch.load(model_path, weights_only=False)['MODEL']
    
    tokenizer = MTTokenizer()
    
    dataset = MTDataset(data_path=args.data)
    data = prepare_dataloader(dataset)
    
    return model, tokenizer, data

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
        num_workers=1
    )


def _hook_metadata(module, inputs, output):
    meta_in = inputs[0].detach()
    module.saved_metadata = meta_in
    
    def _capture(grad):
        module.meta_grad = grad.detach()
    inputs[0].register_hook(_capture)
    
    
def get_attn_relevance(model: torch.nn.Module=None):
    # Figure out gradients for LRP on metadata
    meta_encoder = model.meta_encoder
    meta_in = meta_encoder.saved_metadata
    grads = meta_encoder.meta_grad
    
    relevances = (meta_in * grads).float().cpu()
    return relevances
    
    
def process_sequence(batch, model, tokenizer):
    meta_data, seqs = batch
    meta_data = torch.tensor(meta_data, dtype=torch.float32).to(device).requires_grad_()
    
    inputs = tokenizer.encode(seqs).to(device).unsqueeze(0)
 
    seq_length = inputs.size(1)
#     sequence_accumulator = torch.zeros(23, device="cuda")
    relevance_trace = []
    ids = [i for i in range(1, seq_length+1, 100)]

    for i in tqdm(ids, total=len(ids), desc="Sequence"):
        # Pass sequences that are 100 (x21) iterations apart
        outputs = model(inputs[:, :i, :, :], meta_data).squeeze(0)[-1, :, :]

        max_logits, predicted_ids = torch.max(outputs, dim=-1)
        max_logits.backward(max_logits)
#         agg_max_logits = max_logits.sum() # This may be a better approach? Unclear...
#         agg_max_logits.backward()
        
        input_relevances = get_attn_relevance(model=model)
#         sequence_accumulator += input_relevances
        relevance_trace.append((i, input_relevances, predicted_ids.cpu()))
        
        model.zero_grad()
        del max_logits, input_relevances
        torch.cuda.empty_cache()
#     sentence_accumulator = sentence_accumulator / seq_length 
#     return sentence_accumulator
    return relevance_trace
        
        
def obtain_relevance(model, tokenizer, data):
    rel_per_seq = {}
    for i, batch in tqdm(enumerate(data), total=len(data), desc="Test data"):
        sequence_relevance = process_sequence(batch, model, tokenizer)
        rel_per_seq[i] = sequence_relevance
    return rel_per_seq

def main(args):
    print("Loading model, tokenizer, and dataset...")
    model, tokenizer, data = load_objects(args)
    
    model.meta_encoder.register_forward_hook(_hook_metadata)
    
    print("Obtaining relevances for sequences")
    relevance_traces = obtain_relevance(model, tokenizer, data)
    
    save_dir = f"saved_relevance_traces/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_name = args.model_name.split(".")[0].split("/")[-1]
    save_rel_path = f"{save_dir}/{save_name}_relevances.pkl"
    with open(save_rel_path, "wb") as f:
        pickle.dump(relevance_traces, f)
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)