import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm_train import LSTMModel  # Import the model class

# Define features
features = ['cpu_usage', 'memory_usage']

# Hyperparameters
input_size = len(features)
hidden_size = 64
num_layers = 2
output_size = 3

# Load the trained model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("lstm_model.pth", weights_only=True))
model.eval()

# Load the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_data = pd.read_csv("dataset_10.csv") 
scaler.fit(training_data[features])

# Example input sequence
input_sequence = np.array([
    [18.987341772151897, 14.86132882905838],  # Example values for cpu_usage an>
    [30.88235294117647, 14.89354222399034],
    [24.675324675324674, 15.037692729276275],
    [15.59633027522936, 15.09677307581515],
    [21.052631578947366, 15.374571672673737],
    [54.16666666666667, 15.388850228074682],
    [81.66666666666667, 15.546351230854977],
    [61.97183098591549, 15.5696765173598],
    [37.0, 15.77945092129727],
    [74.285714285714285, 15.814654259392272], 
])

# Scale input data
input_sequence_df = pd.DataFrame(input_sequence, columns=features)
scaled_input_sequence = scaler.transform(input_sequence_df)

# Convert to PyTorch tensor
input_tensor = torch.tensor(scaled_input_sequence, dtype=torch.float32).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(input_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

print(f"Predicted Label: {predicted_label}")
