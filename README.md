# Time Series Analysis Environment

A comprehensive environment for implementing and testing various time series analysis techniques using deep learning models.

## Features

- **Model Implementations:**
  - LSTM and GRU models
  - Transformer-based time series forecasting
  - Autoencoder for anomaly detection

- **Utilities:**
  - Time series data processing
  - Model management and evaluation
  - Data normalization and sequence generation

- **Testing:**
  - Unit tests for all model types
  - Test coverage for core functionality

- **Demonstrations:**
  - Jupyter notebooks with example implementations
  - Sample data generation and visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_test_env_TSA.git
   cd ai_test_env_TSA
Set up the environment:
bash
CopyInsert
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Run the tests:
bash
CopyInsert in Terminal
pytest tests/
Usage
Running the Notebooks
Start Jupyter Notebook:
bash
CopyInsert in Terminal
jupyter notebook
Open and run the demonstration notebooks:
notebooks/lstm_gru_demo.ipynb
notebooks/transformer_demo.ipynb
notebooks/anomaly_detection_demo.ipynb
Training Models
Example usage for training an LSTM model:

python
CopyInsert
from models.lstm_gru.lstm_model import LSTMModel
from utils.time_series_utils import create_sequences, normalize_data

# Prepare data
data = ...  # Your time series data
X, y = create_sequences(data, seq_length=20)
X, mean, std = normalize_data(X)

# Initialize model
model = LSTMModel(input_size=10, hidden_size=32, num_layers=2, output_size=1)

# Train model
...  # Implement training loop
Directory Structure
CopyInsert
ai_test_env_TSA/
├── models/
│   ├── lstm_gru/          # LSTM and GRU implementations
│   ├── transformers/      # Transformer-based models
│   └── anomaly_detection/ # Anomaly detection models
├── tests/                 # Unit tests
├── utils/                 # Utility functions
├── notebooks/             # Demonstration notebooks
├── requirements.txt       # Python dependencies
└── Dockerfile             # Container configuration