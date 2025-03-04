import torch
from models.transformers.time_series_transformer import TimeSeriesTransformer

def test_time_series_transformer():
    input_dim = 10
    model_dim = 64
    num_heads = 4
    num_layers = 3
    output_dim = 1
    seq_length = 20
    batch_size = 16
    
    model = TimeSeriesTransformer(
        input_dim, model_dim, num_heads, num_layers, output_dim, seq_length
    )
    x = torch.randn(batch_size, seq_length, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, output_dim), \
        f"Expected shape {(batch_size, output_dim)}, got {output.shape}"