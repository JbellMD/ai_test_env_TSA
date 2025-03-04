import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, seq_len, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.positional_encoding = self._init_positional_encoding(seq_len, model_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(model_dim, output_dim)
    
    def _init_positional_encoding(self, seq_len, model_dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(seq_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:x.size(1), :].to(x.device)
        
        # Transformer encoder
        x = x.permute(1, 0, 2)  # (seq_len, batch, model_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, model_dim)
        
        # Output projection
        output = self.output_proj(x[:, -1, :])
        return output