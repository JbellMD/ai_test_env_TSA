import torch
from models.lstm_gru.lstm_model import LSTMModel
from models.lstm_gru.gru_model import GRUModel

def test_lstm_model():
    input_size = 10
    hidden_size = 32
    num_layers = 2
    output_size = 1
    batch_size = 16
    seq_length = 20
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    x = torch.randn(batch_size, seq_length, input_size)
    output = model(x)
    
    assert output.shape == (batch_size, output_size), \
        f"Expected shape {(batch_size, output_size)}, got {output.shape}"

def test_gru_model():
    input_size = 10
    hidden_size = 32
    num_layers = 2
    output_size = 1
    batch_size = 16
    seq_length = 20
    
    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    x = torch.randn(batch_size, seq_length, input_size)
    output = model(x)
    
    assert output.shape == (batch_size, output_size), \
        f"Expected shape {(batch_size, output_size)}, got {output.shape}"