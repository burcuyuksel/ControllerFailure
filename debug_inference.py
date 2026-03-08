import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out

# Deneme verisi (İdeal boşta/idle değerleri)
# features = ['cpu_usage', 'memory_usage', 'heap_mb', 'heap_growth_bps', 'proc_cpu_pct', 'rest_rtt_ms']
idle_data = np.array([
    [10.0, 45.0, 110.0, 0.0, 5.0, 5.0],
    [10.1, 45.1, 111.0, 100.0, 5.5, 5.2],
    [10.2, 45.1, 112.0, 100.0, 5.5, 5.2],
    [10.5, 45.2, 113.0, 100.0, 5.1, 5.0],
    [10.0, 45.0, 110.0, 0.0, 5.0, 5.0],
    [10.1, 45.1, 111.0, 100.0, 5.5, 5.2],
    [10.2, 45.1, 112.0, 100.0, 5.5, 5.2],
    [10.5, 45.2, 113.0, 100.0, 5.1, 5.0],
    [10.0, 45.0, 110.0, 0.0, 5.0, 5.0],
    [10.1, 45.1, 111.0, 100.0, 5.5, 5.2],
])

scaler = joblib.load("scaler.pkl")
print("Scaler Mins (0 değerine karşılık gelen):", scaler.data_min_)
print("Scaler Maxs (1 değerine karşılık gelen):", scaler.data_max_)

scaled_idle = scaler.transform(idle_data)
print("\nScaled Idle Input:\n", scaled_idle[-1])

model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, output_size=3)
model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

# Sequence halini ver
X = torch.tensor([scaled_idle], dtype=torch.float32)
with torch.no_grad():
    out = model(X)
    probs = torch.softmax(out, dim=1)[0]

print(f"\nProbs - N: {probs[0]:.3f}, S: {probs[1]:.3f}, F: {probs[2]:.3f}")
