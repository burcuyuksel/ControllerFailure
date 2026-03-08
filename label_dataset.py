"""
label_dataset.py
────────────────────────────────────────────────────────────
7MartTest.txt'teki 13 testten c1 controller metriklerini çıkar,
kullanıcının belirlediği stress/fail döngü numaralarına göre
NORMAL / STRESS / FAIL etiketle ve CSV'e kaydet.
────────────────────────────────────────────────────────────
"""

import re
import csv
import sys

# ── Test sınır noktaları (döngü numarası, dahil) ──────────
# (stress_start, fail_start)
# stress_start: bu döngüden itibaren STRESS
# fail_start  : bu döngüden itibaren FAIL
TEST_BOUNDARIES = [
    (20, 30),   # Test 1
    (16, 25),   # Test 2
    (13, 21),   # Test 3
    (11, 19),   # Test 4
    (11, 18),   # Test 5
    (10, 17),   # Test 6
    (11, 20),   # Test 7
    (11, 24),   # Test 8
    (12, 20),   # Test 9
    ( 8, 16),   # Test 10
    (11, 19),   # Test 11
    (12, 24),   # Test 12   ("12 fail" → stress@12 kabul edildi
    (16, 23),   # Test 13
]

INPUT_FILE  = "7MartTest.txt"
OUTPUT_FILE = "dataset_7mart.csv"

# ── Satır formatı: "    5 | 2026-03-07 15:13:03 |   c1 | ..."
# Grup: (seq#, timestamp, ctrl, cpu, mem, heap, hgrow, pcpu, rtt)
LINE_RE = re.compile(
    r"^\s+(\d+)\s+\|\s+([\d\- :]+)\s+\|\s+(\w+)\s+\|"
    r"\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+|-)\s+\|"
    r"\s+([-\d]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)"
)

# lstm_train.py'ın beklediği sayısal etiketler
LABEL_MAP = {"NORMAL": 0, "STRESS": 1, "FAIL": 2}

def label(seq, stress_start, fail_start):
    if seq >= fail_start:
        return LABEL_MAP["FAIL"]
    elif seq >= stress_start:
        return LABEL_MAP["STRESS"]
    else:
        return LABEL_MAP["NORMAL"]

def parse_and_label(input_path, output_path):
    rows = []
    test_idx  = -1   # hangi test bloğundayız
    in_block  = False

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            # Yeni test bloğu başlığı (header çizgisi)
            if "Floodlight 3-Controller Metrik" in line:
                test_idx += 1
                in_block = True
                print(f"[Test {test_idx + 1}] blok başladı")
                continue

            if not in_block:
                continue

            m = LINE_RE.match(line)
            if not m:
                continue

            seq_no, ts, ctrl, cpu, mem, heap, hgrow, pcpu, rtt = m.groups()

            # Sadece c1 satırlarını işle
            if ctrl.strip() != "c1":
                continue

            seq_no = int(seq_no)

            if test_idx >= len(TEST_BOUNDARIES):
                print(f"[UYARI] Test {test_idx+1} için sınır tanımlanmamış, atlandı.")
                continue

            s_start, f_start = TEST_BOUNDARIES[test_idx]
            lbl = label(seq_no, s_start, f_start)

            rows.append({
                "timestamp"      : ts.strip(),
                "cpu_usage"      : cpu,
                "memory_usage"   : mem,
                "heap_mb"        : 0 if heap == "-" else heap,
                "heap_growth_bps": hgrow,
                "proc_cpu_pct"   : pcpu,
                "rest_rtt_ms"    : rtt,
                "label"          : lbl,
            })

    # CSV yaz
    fieldnames = ["timestamp","cpu_usage","memory_usage","heap_mb",
                  "heap_growth_bps","proc_cpu_pct","rest_rtt_ms","label"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Özet
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    print(f"\n✓ {len(rows)} satır yazıldı → {output_path}")
    print(f"  NORMAL (0): {counts[0]}")
    print(f"  STRESS (1): {counts[1]}")
    print(f"  FAIL   (2): {counts[2]}")



if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    parse_and_label(inp, out)
