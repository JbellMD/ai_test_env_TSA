import torch
from models.anomaly_detection.autoencoder import AnomalyDetector

def test_anomaly_detector():
    input_dim = 10
    encoding_dim = 3
    batch_size = 16
    
    model = AnomalyDetector(input_dim, encoding_dim)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, input_dim), \
        f"Expected shape {(batch_size, input_dim)}, got {output.shape}"