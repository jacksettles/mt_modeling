import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
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


class MTLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, num_classes=4, embed_dim=8):
        super(MTLSTM, self).__init__()
        
        self.meta_encoder = MetadataEncoder(output_dim=260*embed_dim)
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.lstm = nn.LSTM(input_size = 260 * embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 260 * num_classes)

    def forward(self, x, meta_data):
        """
        Args:
            x: Tensor of shape (batch, seq_length, 260, 1)
            meta_data: Tensor of shape (23), or number of simulator features
        """
        B, L, N, _ = x.shape      # B=batch, L=seq_len, N=260
        x = x.squeeze(-1).long()  # [B, L, 260]
        x = self.embedding(x)     # [B, L, 260, 8]
        x = x.view(B, L, -1)      # [B, L, 2080]
        
        encoded_meta = self.meta_encoder(meta_data).unsqueeze(0).unsqueeze(0)
        
        x = torch.cat([encoded_meta, x], dim=1) # [B, L+1, hidden_dim]
        
        lstm_out, _ = self.lstm(x)  # [B, L+1, hidden_dim]
        out = self.fc(lstm_out)     # [B, L+1, 1040]
        out = out.view(B, L+1, 260, 4)  # [B, L, 260, 4] — logits per position
        return out