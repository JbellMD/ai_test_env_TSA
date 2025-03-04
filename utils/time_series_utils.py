import numpy as np
import pandas as pd

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8), mean, std

def denormalize_data(data, mean, std):
    return data * (std + 1e-8) + mean

def split_data(data, train_ratio=0.8):
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]