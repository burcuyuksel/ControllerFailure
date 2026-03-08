import time
import json
from collect_metrics import collect_all_controllers, CONTROLLERS

# Configuration
log_file = "real_time_metrics.log"

def fmt(val, width=7, decimals=1):
    if val is None:
        return "-".rjust(width)
    if isinstance(val, float):
        return f"{val:{width}.{decimals}f}"
    return str(val).rjust(width)

print("=" * 90)
print("  Floodlight 3-Controller Metrik Toplama")
print(f"  Log: {log_file} | Ctrl+C ile durdur")
print("=" * 90)

header = f"{'#':>5} | {'Zaman':>19} | {'Ctrl':>4} | {'CPU%':>6} | {'Mem%':>6} | {'HeapMB':>7} | {'HGrow':>8} | {'pCPU%':>6} | {'RTT':>7}"
print(header)
print("-" * 90)

count = 0
while True:
    all_metrics = collect_all_controllers()
    if all_metrics:
        with open(log_file, "a") as log:
            log.write(json.dumps(all_metrics) + "\n")
        count += 1

        for name in ["c1", "c2", "c3"]:
            m = all_metrics.get(name)
            if not m:
                print(f"{count:5} | {'':>19} | {name:>4} | DOWN")
                continue

            heap_mb = f"{m['heap_used_bytes']/1e6:7.1f}" if m.get('heap_used_bytes') else "      -"
            hgrow = fmt(m.get('heap_growth_bps'), 8, 0) if m.get('heap_growth_bps') is not None else "       -"

            print(
                f"{count:5} | {m['timestamp']:>19} | {name:>4} | "
                f"{fmt(m.get('cpu_usage'), 6)} | {fmt(m.get('memory_usage'), 6)} | {heap_mb} | {hgrow} | "
                f"{fmt(m.get('proc_cpu_pct'), 6)} | {fmt(m.get('rest_rtt_ms'), 7)}"
            )
    else:
        count += 1
        print(f"{count:5} | Bağlantı hatası...")

    time.sleep(0.5)
