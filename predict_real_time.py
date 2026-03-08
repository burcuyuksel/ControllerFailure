import torch
import numpy as np
from lstm_train import LSTMModel

# Define paths
MODEL_PATH = "lstm_model.pth"

# Model architecture MUST match lstm_train.py exactly
input_size = 6        # cpu_usage, memory_usage, heap_mb, heap_growth_bps, proc_cpu_pct, rest_rtt_ms
hidden_size = 64
num_layers = 2
output_size = 3       # 0: Normal, 1: Stress, 2: Fail
dropout = 0.3

# Initialize model
try:
    model = LSTMModel(input_size=input_size, 
                      hidden_size=hidden_size, 
                      num_layers=num_layers, 
                      output_size=output_size, 
                      dropout=dropout)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Real-time Prediction Model loaded successfully.")
except FileNotFoundError:
    print(f"WARNING: Model file {MODEL_PATH} not found!")

def predict_real_time(X_real_time):
    """
    X_real_time : shape [1, sequence_length, num_features]
    Returns list of (predicted_label, stress_prob, fail_prob) tuples.
      predicted_label : 0=Normal, 1=Stress, 2=Fail
      stress_prob     : P(Stress) — multi-signal karar için kullanılır
      fail_prob       : P(Fail)
    """
    if len(X_real_time) == 0:
        return []

    with torch.no_grad():
        sequence_tensor = torch.tensor(X_real_time)
        output = model(sequence_tensor)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        norm_prob   = probabilities[0][0].item()
        stress_prob = probabilities[0][1].item()
        fail_prob   = probabilities[0][2].item()

        # Eşik bazlı etiket kararı (0.60 üstü kesin sınıf)
        if fail_prob > 0.60:
            predicted_label = 2
        elif stress_prob > 0.60:
            predicted_label = 1
        else:
            _, predicted_label_tensor = torch.max(output, 1)
            predicted_label = predicted_label_tensor.item()

        print(f"Probs - Normal: {norm_prob:.3f}, Stress: {stress_prob:.3f}, Fail: {fail_prob:.3f}")

    return [(predicted_label, stress_prob, fail_prob)]

if __name__ == "__main__":
    from preprocess_real_time import preprocess_real_time
    X = preprocess_real_time("real_time_metrics.log")
    if len(X) > 0:
        pred = predict_real_time(X)
        labels = ["Normal", "Stress", "Fail"]
        print(f"Prediction result: {pred[0]} ({labels[pred[0]]})")
    else:
        print("Not enough data to predict.")