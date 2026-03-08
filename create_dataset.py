"""
real_time_metrics.log (JSON-lines) dosyasından c1 verilerini okuyup
kullanıcının belirttiği faz geçişlerine göre etiketleyerek CSV dataset oluşturur.

Label:
  0 = Normal / Open (recovery)
  1 = Stress
  2 = Fail (controller down)
"""
import json
import csv

LOG_FILE = "real_time_metrics.log"
OUTPUT_FILE = "dataset_c1.csv"
CONTROLLER = "c1"

# Kullanıcının belirttiği faz geçişleri (ölçüm numarası, yeni_faz)
# Faz: "normal", "stress", "fail", "open"
PHASE_TRANSITIONS = [
    (1,   "normal"),
    (236, "stress"),
    (248, "open"),
    (262, "stress"),
    (268, "fail"),
    (280, "open"),
    (294, "stress"),
    (301, "fail"),
    (312, "open"),
    (323, "stress"),
    (329, "fail"),
    (344, "open"),
    (355, "stress"),
    (360, "fail"),
    (383, "open"),
    (394, "stress"),
    (402, "fail"),
    (411, "open"),
    (446, "stress"),
    (452, "fail"),
    (462, "open"),
    (470, "stress"),
    (475, "fail"),
    (491, "open"),
    (505, "stress"),
    (510, "fail"),
    (520, "open"),
    (526, "stress"),
    (531, "fail"),
    (538, "open"),
    (545, "stress"),
    (550, "fail"),
    (558, "open"),
    (566, "stress"),
]

LABEL_MAP = {"normal": 0, "open": 0, "stress": 1, "fail": 2}

def get_label(measurement_num):
    """Ölçüm numarasına göre label döndür."""
    current_phase = "normal"
    for start, phase in PHASE_TRANSITIONS:
        if measurement_num >= start:
            current_phase = phase
        else:
            break
    return LABEL_MAP[current_phase]

# CSV sütunları
COLUMNS = ["timestamp", "cpu_usage", "memory_usage", "heap_used_bytes",
           "heap_growth_bps", "proc_cpu_pct", "rest_rtt_ms", "label"]

def safe_val(val):
    """None → 0 dönüştür."""
    if val is None:
        return 0.0
    return val

# Log dosyasını oku ve CSV oluştur
with open(LOG_FILE, "r") as f:
    lines = f.readlines()

rows = []
for i, line in enumerate(lines):
    measurement_num = i + 1
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        continue

    ctrl_data = data.get(CONTROLLER)
    if not ctrl_data:
        continue

    label = get_label(measurement_num)

    row = {
        "timestamp": ctrl_data.get("timestamp", ""),
        "cpu_usage": safe_val(ctrl_data.get("cpu_usage")),
        "memory_usage": safe_val(ctrl_data.get("memory_usage")),
        "heap_used_bytes": safe_val(ctrl_data.get("heap_used_bytes")),
        "heap_growth_bps": safe_val(ctrl_data.get("heap_growth_bps")),
        "proc_cpu_pct": safe_val(ctrl_data.get("proc_cpu_pct")),
        "rest_rtt_ms": safe_val(ctrl_data.get("rest_rtt_ms")),
        "label": label,
    }
    rows.append(row)

# CSV yaz
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

# Özet
label_counts = {0: 0, 1: 0, 2: 0}
for r in rows:
    label_counts[r["label"]] += 1

print(f"Dataset oluşturuldu: {OUTPUT_FILE}")
print(f"Toplam satır: {len(rows)}")
print(f"  Label 0 (Normal/Open): {label_counts[0]}")
print(f"  Label 1 (Stress):      {label_counts[1]}")
print(f"  Label 2 (Fail):        {label_counts[2]}")
