import json
import pandas as pd
import numpy as np
import joblib

# Load the scaler saved during training (so we use the EXACT same min-max values)
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("WARNING: scaler.pkl not found! Please run lstm_train.py first.")
    scaler = None

# Features must be in the exact same order as trained
FEATURES = ['cpu_usage', 'memory_usage', 'heap_mb', 'heap_growth_bps', 'proc_cpu_pct', 'rest_rtt_ms']
CONTROLLER_DEFAULT = "c1"

def preprocess_real_time(log_file, sequence_length=10):
    if scaler is None:
        return np.array([])

    with open(log_file, "r") as file:
        lines = file.readlines()
        
    # Sadece JSON formatındaki ve c1'e ait son N satırı al
    # Hızlı olması için tüm dosyayı parse etmek yerine, sondan okuma yapılabilir.
    parsed_data = []
    for line in lines[-50:]:  # Sadece son 50 satıra bakmak performansı artırır
        if not line.strip(): continue
        try:
            data = json.loads(line.strip())
            
            # Eğer format c1: {...} şeklindeyse (yeni format):
            if CONTROLLER_DEFAULT in data:
                ctrl_data = data[CONTROLLER_DEFAULT]
            elif 'cpu_usage' in data: # Eski format
                ctrl_data = data
            else:
                continue

            row = {
                "cpu_usage": ctrl_data.get("cpu_usage", 0.0) or 0.0,
                "memory_usage": ctrl_data.get("memory_usage", 0.0) or 0.0,
                "heap_mb": (ctrl_data.get("heap_used_bytes", 0.0) or 0.0) / (1024.0 * 1024.0),
                "heap_growth_bps": ctrl_data.get("heap_growth_bps", 0.0) or 0.0,
                "proc_cpu_pct": ctrl_data.get("proc_cpu_pct", 0.0) or 0.0,
                "rest_rtt_ms": ctrl_data.get("rest_rtt_ms", 0.0) or 0.0
            }
            parsed_data.append(row)
        except json.JSONDecodeError:
            continue

    if len(parsed_data) < sequence_length:
        return np.array([])  # Not enough data for a sequence yet

    # DataFrame'e çevirip normalize et
    df = pd.DataFrame(parsed_data)
    
    # Bazı anahtarlar JSON'dan farklı gelebilir: 'heap_used_bytes' olarak gelirse
    # 'heap_mb' sütunu olarak isimlendirip kullanmalıyız.
    
    # Extract only the needed features in exactly the right order
    feature_df = df[FEATURES]
    
    # Normalize using the pre-fitted scaler
    scaled_data = scaler.transform(feature_df)
    
    # We only need the prediction for the VERY LAST sequence
    # Son 10 zaman adımını (sequence) alalım
    recent_sequence = scaled_data[-sequence_length:]
    
    # Model expects shape (batch_size, sequence_length, features)
    # Return one sequence: shape (1, 10, 6)
    return np.array([recent_sequence], dtype=np.float32)

if __name__ == "__main__":
    X_rt = preprocess_real_time("real_time_metrics.log")
    print("Test Output Shape:", X_rt.shape)
