import pandas as pd
import re

# 1. Read raw text dump
with open("new_dataset.txt", "r") as f:
    lines = f.readlines()

# Patterns
# 1 | 2026-03-02 23:13:27 |   c1 |    0.0 |   44.7 |   116.2 |        0 |   44.5 |     7.7
log_pattern = re.compile(r'^\s*(\d+)\s*\|\s*([\d\-:\s]+)\s*\|\s*(c\d)\s*\|\s*([0-9\.\-]+)\s*\|\s*([0-9\.\-]+)\s*\|\s*([0-9\.\-]+)\s*\|\s*([0-9\.\-]+)\s*\|\s*([0-9\.\-]+)\s*\|\s*([0-9\.\-]+)')

parsed_data = []

current_sequence_id = 0

for line in lines:
    match = log_pattern.match(line)
    if match:
        seq_id = int(match.group(1))
        ts = match.group(2).strip()
        ctrl = match.group(3).strip()
        cpu = match.group(4).strip()
        mem = match.group(5).strip()
        heap = match.group(6).strip()
        hgrow = match.group(7).strip()
        pcpu = match.group(8).strip()
        rtt = match.group(9).strip()
        
        if ctrl == "c1":  # We only care about c1 for the model right now
            try:
                row = {
                    "sequence_id": seq_id,
                    "timestamp": ts,
                    "cpu_usage": float(cpu) if cpu != "-" else 0.0,
                    "memory_usage": float(mem) if mem != "-" else 0.0,
                    "heap_mb": float(heap) if heap != "-" else 0.0,
                    "heap_growth_bps": float(hgrow) if hgrow != "-" else 0.0,
                    "proc_cpu_pct": float(pcpu) if pcpu != "-" else 0.0,
                    "rest_rtt_ms": float(rtt) if rtt != "-" else 0.0,
                }
                parsed_data.append(row)
            except ValueError:
                pass


df = pd.DataFrame(parsed_data)

# 2. Labeling Logic based on user's definition.
# 8 stres, 21 fail, 33 open, 45 stres, 57 fail, 69 open, 82 stres, 95 fail...
# 0: Open/Normal, 1: Stress, 2: Fail

def get_label(seq_id):
    if seq_id < 8: return 0
    if seq_id < 21: return 1
    if seq_id < 33: return 2
    if seq_id < 45: return 0
    if seq_id < 57: return 1
    if seq_id < 69: return 2
    if seq_id < 82: return 0
    if seq_id < 95: return 1
    if seq_id < 111: return 2
    if seq_id < 124: return 0
    if seq_id < 136: return 1
    if seq_id < 147: return 2
    if seq_id < 163: return 0
    if seq_id < 177: return 1
    if seq_id < 188: return 2
    if seq_id < 200: return 0
    if seq_id < 212: return 1
    if seq_id < 222: return 2
    if seq_id < 244: return 0
    if seq_id < 253: return 1
    return 0

df['label'] = df['sequence_id'].apply(get_label)

# Select only desired columns (drop seq_id to match our feature set format)
df_final = df[['timestamp', 'cpu_usage', 'memory_usage', 'heap_mb', 'heap_growth_bps', 'proc_cpu_pct', 'rest_rtt_ms', 'label']]

df_final.to_csv("dataset_c1_v2.csv", index=False)
print("Data saved to dataset_c1_v2.csv")
