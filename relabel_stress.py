"""
relabel_stress.py
─────────────────────────────────────────────────────────────
dataset_7mart.csv'deki NORMAL/STRESS sınırını, curl basma anına
göre değil metriklerin gerçekten değiştiği ana göre otomatik
yeniden belirler.

Kural:
  FAIL   → memory_usage == 0 (değişmez)
  STRESS → test başındaki heap ortalamasının %40 üzerine çıkan
            ilk andan itibaren (bir kez tetiklendi mi, hep STRESS)
  NORMAL → geri kalan
─────────────────────────────────────────────────────────────
"""

import pandas as pd
from collections import Counter

INPUT  = "dataset_7mart.csv"
OUTPUT = "dataset_7mart.csv"   # üzerine yazar

STRESS_RATIO = 1.40   # baseline'ın kaç katında stress başlar

df = pd.read_csv(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Test gruplarını tespit et (>30s zaman boşluğu = yeni test)
time_diffs = df["timestamp"].diff().dt.total_seconds().fillna(0)
df["_group"] = (time_diffs > 30).cumsum()

new_labels = []

for gid, group in df.groupby("_group", sort=True):
    group = group.copy()

    # Fail satırları: memory_usage == 0
    fail_mask = group["memory_usage"] == 0.0
    non_fail  = group[~fail_mask]

    # Baseline heap: gruptaki ilk 4 non-fail satırın ortalaması
    baseline_heap = non_fail["heap_mb"].head(4).mean()
    stress_thresh = baseline_heap * STRESS_RATIO

    labels      = []
    in_stress   = False   # bir kez stress tetiklendiyse hep stress

    for _, row in group.iterrows():
        if row["memory_usage"] == 0.0:
            labels.append(2)  # FAIL
        else:
            if not in_stress:
                if row["heap_mb"] > stress_thresh:
                    in_stress = True
            labels.append(1 if in_stress else 0)

    new_labels.extend(labels)

df["label"] = new_labels
df = df.drop(columns=["_group"])
df.to_csv(OUTPUT, index=False)

# Özet
counts = Counter(new_labels)
print(f"✓ {len(df)} satır yeniden etiketlendi → {OUTPUT}")
print(f"  NORMAL (0): {counts[0]}")
print(f"  STRESS (1): {counts[1]}")
print(f"  FAIL   (2): {counts[2]}")

# Test bazlı ilk stress satırı
df["_group"] = (df["timestamp"].diff().dt.total_seconds().fillna(0) > 30).cumsum()
print("\nTest bazlı stress başlangıcı:")
for gid, g in df.groupby("_group"):
    stress_rows = g[g["label"] == 1]
    fail_rows   = g[g["label"] == 2]
    s_ts = stress_rows["timestamp"].iloc[0].strftime("%H:%M:%S") if len(stress_rows) else "-"
    f_ts = fail_rows["timestamp"].iloc[0].strftime("%H:%M:%S")   if len(fail_rows)   else "-"
    baseline = g[g["memory_usage"] > 0]["heap_mb"].head(4).mean()
    thresh   = baseline * STRESS_RATIO
    print(f"  Test {gid+1:2d}: baseline={baseline:.0f}MB  eşik={thresh:.0f}MB  "
          f"stress@{s_ts}  fail@{f_ts}  "
          f"[N={len(g[g['label']==0])} S={len(stress_rows)} F={len(fail_rows)}]")
