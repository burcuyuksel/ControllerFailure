import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = pd.read_csv("dataset_10.csv")  # Adjust filename accordingly

# Feature selection
features = ['cpu_usage', 'memory_usage']
labels = ['label']

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features])

# Add labels back to the dataset
data_scaled = pd.DataFrame(scaled_features, columns=features)
data_scaled['label'] = data['label']

# Create sequences for LSTM
sequence_length = 10  # Number of timesteps in each sequence
X, y = [], []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled.iloc[i:i + sequence_length][features].values)
    y.append(data_scaled.iloc[i + sequence_length]['label'])

X = np.array(X)
y = np.array(y)
