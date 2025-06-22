import torch
import torch.nn as nn
import math

def generate_causal_mask(seq_length, batch_size, num_heads):
    """
    Generates a causal mask to prevent states from attending to future states.
    """
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)  # Upper triangular mask
    mask = mask.masked_fill(mask == 1, float('-inf'))  # Convert to attention mask
    mask = mask.unsqueeze(0)
    mask = mask.expand(batch_size*num_heads, seq_length, seq_length)  # Expand to (batch_size * num_heads, seq_length, seq_length)
    return mask


class MTTokenizer():
    """
    For converting MT sequences from a list of strings to tensors of integer IDs.
    """
    def __init__(self, vocab=['O', 'G', 'X', 'A']):
        self.vocab = vocab
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(vocab)}
        
    def encode(self, batch, return_tensors=True):
        encoded_list =  [[[self.char2idx[char] for char in state] for state in sequence] for sequence in batch]
        if return_tensors:
            return torch.tensor(encoded_list, dtype=torch.long)
        else:
            return encoded_list
    
    def decode(self, encoded_state):
        return "".join([self.idx2char[idx.item()] for idx in encoded_state])
    
    
class MetadataEncoder(nn.Module):
    """
    To encode the metadata into the same dimensionality as the MT states in the sequence.
    """
    def __init__(self, input_dim=23, output_dim=2080):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 3072),
            torch.nn.ReLU(),
            torch.nn.Linear(3072, output_dim)
        )
        self.output_dim = output_dim
        
    def forward(self, meta):
        out = self.mlp(meta)
        return out
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2380, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings once up to max_len
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()        # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))      # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos * div_term)                   # even dims
        pe[:, 1::2] = torch.cos(pos * div_term)                   # odd dims
        pe = pe.unsqueeze(0)                                       # (1, max_len, d_model)

        # Register as buffer so it's on the right device and not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
          x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
          x with positional encodings added, same shape.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
    
    
class MTStatePredictor(nn.Module):
    """
    Transformer based MT sequence model.
    """
    def __init__(self, vocab_size=4, state_length=260, embed_dim=8, num_heads=8, num_layers=6, hidden_dim=2048, device="cuda:0"):
        super().__init__()
        self.num_heads = num_heads
        self.state_length = state_length
        self.embed_dim = embed_dim
        
        self.meta_encoder = MetadataEncoder(output_dim=state_length*embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(d_model=state_length*embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(state_length*embed_dim, num_heads, hidden_dim, norm_first=True, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(state_length*embed_dim, state_length*embed_dim) 
        self.device = device
        self.to(self.device)
        
        
    def forward(self, x, metadata):
        """
        Args:
            x: Tensor of shape (batch, seq_length, 260, 1)
            metadata: Tensor of shape (23), or number of simulator features
        """
#         print(f"X shape: {x.shape}")
        B, L, state_length, char_dim = x.shape
        x = self.embedding(x)
        x = x.view(B, L, -1)
        encoded_metadata = self.meta_encoder(metadata).unsqueeze(0).unsqueeze(0)
#         print(f"Metadata: {encoded_metadata.shape}")
#         print(f"X shape: {x.shape}")

        x = torch.cat([encoded_metadata, x], dim=1)
        x = self.positional_encoding(x)
        
        causal_mask = generate_causal_mask(x.size(1), x.size(0), self.num_heads).to(self.device)

        x = self.encoder(x, mask=causal_mask)
        x = self.fc(x)
        x = x.view(B, L+1, self.state_length, self.embed_dim)
        logits = torch.matmul(x, self.embedding.weight.T) # using embedding weights to map directly back to vocab
        return logits