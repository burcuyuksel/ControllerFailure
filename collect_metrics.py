import requests
import subprocess
from datetime import datetime
import time
import re
import os

# Configuration - 3 Floodlight Controller (Aynı VM içerisinde)
CONTROLLERS = [
    {"name": "c1", "ip": "192.168.56.107", "rest_port": "8080", "of_port": "6653"},
    {"name": "c2", "ip": "192.168.56.107", "rest_port": "8081", "of_port": "7753"},
    {"name": "c3", "ip": "192.168.56.107", "rest_port": "8082", "of_port": "8853"},
]

_prev = {}  # Delta hesapları için
_last_cpu = {"idle": 0, "total": 0}

def local_exec(cmd, timeout=1.5):
    """Aynı makine içerisinde local komut çalıştır ve stdout döndür."""
    # İşletim sistemi seviyesinde de timeout zorlaması (Komut takılırsa OS öldürsün)
    safe_cmd = f"timeout {timeout} {cmd}"
    try:
        result = subprocess.run(safe_cmd, shell=True, capture_output=True, text=True, timeout=timeout+0.5)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def get_system_cpu_pct():
    """Sistem geneli CPU (Uyuma / Sleep OLMADAN stateless hesaplama)."""
    try:
        with open('/proc/stat', 'r') as f:
            line = f.readline()
        
        vals = list(map(int, line.split()[1:]))
        idle = vals[3]
        total = sum(vals)
        
        prev_idle = _last_cpu["idle"]
        prev_total = _last_cpu["total"]
        
        _last_cpu["idle"] = idle
        _last_cpu["total"] = total
        
        if prev_total == 0:
            return 0.0 # İlk okuma
            
        diff_idle = idle - prev_idle
        diff_total = total - prev_total
        if diff_total == 0:
            return 0.0
            
        cpu_pct = (1.0 - (float(diff_idle) / diff_total)) * 100.0
        return round(max(cpu_pct, 0.0), 1)
    except Exception:
        return 0.0

def get_floodlight_pid(rest_port):
    """Belirli bir REST portunu dinleyen uygulamanın PID'sini bulur."""
    out = local_exec(f"ss -lptn | grep ':{rest_port} '", timeout=1.0)
    if out:
        match = re.search(r'pid=(\d+)', out)
        if match:
            return int(match.group(1))
            
    out2 = local_exec(f"lsof -t -i:{rest_port}", timeout=1.0)
    if out2 and out2.strip().isdigit():
        return int(out2.strip())
        
    return None

def get_process_metrics(pid):
    """Process içi metrikler, doğrudan Linux'un kendisinden istenir."""
    res = {"proc_cpu_pct": 0.0, "proc_rss_kb": 0, "threads": 0, "fd_count": 0}
    if not pid:
        return res

    ps_out = local_exec(f"ps -p {pid} -o %cpu=,rss=,nlwp=", timeout=1.0)
    if ps_out:
        parts = ps_out.split()
        try:
            if len(parts) >= 1: res["proc_cpu_pct"] = float(parts[0])
            if len(parts) >= 2: res["proc_rss_kb"] = int(float(parts[1]))
            if len(parts) >= 3: res["threads"] = int(float(parts[2]))
        except Exception:
            pass

    try:
        fd_count = len(os.listdir(f"/proc/{pid}/fd"))
        res["fd_count"] = fd_count
    except Exception:
        pass

    return res

def get_gc_metrics_jstat(pid):
    """GC metrikleri jstat komutuyla JVM'in içinden alınır."""
    res = {"ygc": 0.0, "fgc": 0.0, "ygct": 0.0, "fgct": 0.0}
    if not pid:
        return res

    out = local_exec(f"jstat -gc {pid} 1000 1 2>/dev/null", timeout=1.5)
    if not out:
        return res

    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return res
        
    header = lines[0].split()
    vals = lines[1].split()

    try:
        res["ygc"] = float(vals[header.index("YGC")])
        res["ygct"] = float(vals[header.index("YGCT")])
        res["fgc"] = float(vals[header.index("FGC")])
        res["fgct"] = float(vals[header.index("FGCT")])
    except Exception:
        pass

    return res

def _delta_features(ctrl_name, now_ts, heap_used_bytes, gc):
    """Önceki state ile delta/türev hesapla (Memory Growth BPS vb.)"""
    out = {
        "heap_growth_bps": 0.0,
        "ygc_per_s": 0.0,
        "fgc_per_s": 0.0,
        "gc_overhead_ratio": 0.0,
    }

    prev = _prev.get(ctrl_name)
    if not prev:
        _prev[ctrl_name] = {
            "ts": now_ts,
            "heap_used_bytes": heap_used_bytes,
            "ygc": gc.get("ygc"),
            "fgc": gc.get("fgc"),
            "ygct": gc.get("ygct"),
            "fgct": gc.get("fgct"),
        }
        return out

    dt = (now_ts - prev["ts"])
    if dt <= 0:
        dt = 1e-6

    if heap_used_bytes is not None and prev.get("heap_used_bytes") is not None:
        out["heap_growth_bps"] = (heap_used_bytes - prev["heap_used_bytes"]) / dt

    if gc.get("ygc") is not None and prev.get("ygc") is not None:
        out["ygc_per_s"] = (gc["ygc"] - prev["ygc"]) / dt
    if gc.get("fgc") is not None and prev.get("fgc") is not None:
        out["fgc_per_s"] = (gc["fgc"] - prev["fgc"]) / dt

    if (gc.get("ygct") is not None and gc.get("fgct") is not None and
        prev.get("ygct") is not None and prev.get("fgct") is not None):
        d_gc_time = (gc["ygct"] - prev["ygct"]) + (gc["fgct"] - prev["fgct"])
        out["gc_overhead_ratio"] = max(0.0, d_gc_time / dt)

    _prev[ctrl_name] = {
        "ts": now_ts,
        "heap_used_bytes": heap_used_bytes,
        "ygc": gc.get("ygc"),
        "fgc": gc.get("fgc"),
        "ygct": gc.get("ygct"),
        "fgct": gc.get("fgct"),
    }
    return out

def collect_metrics_for(controller):
    """Tek bir controller'dan metrik topla (LSTM feature set)."""
    name = controller["name"]
    ip = controller["ip"]
    rest_port = controller["rest_port"]
    memory_url = f"http://{ip}:{rest_port}/wm/core/memory/json"

    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_ts = time.time()

    rest_ok = 0
    rest_rtt_ms = 0.0
    total_mem = 0
    free_mem = 0
    heap_used_bytes = 0
    heap_used_ratio = 0.0
    memory_usage_pct = 0.0

    try:
        t0 = time.time()
        # Eğer Controller %100 CPU ise ve cevap veremiyorsa hemen pes et ki sistem kilitlenmesin (Timeout=1.5 sn)
        mem_response = requests.get(memory_url, timeout=1.5)
        rest_rtt_ms = (time.time() - t0) * 1000.0
        mem_response.raise_for_status()
        mem_data = mem_response.json()
        rest_ok = 1

        total_mem = int(mem_data.get('total', 0) or 0)
        free_mem = int(mem_data.get('free', 0) or 0)
    except requests.exceptions.Timeout:
        rest_rtt_ms = 1500.0  # Madem timeout oldu (pes etti), maksimum değer 1.5 sn olarak girilsin
    except Exception:
        pass

    if total_mem and total_mem > 0:
        heap_used_bytes = max(total_mem - (free_mem or 0), 0)
        heap_used_ratio = heap_used_bytes / total_mem
        memory_usage_pct = heap_used_ratio * 100.0

    cpu_usage = get_system_cpu_pct()
    pid = get_floodlight_pid(rest_port)
    
    proc = get_process_metrics(pid)
    gc = get_gc_metrics_jstat(pid)
    deltas = _delta_features(name, now_ts, heap_used_bytes, gc)

    return {
        "timestamp": ts_str,
        "cpu_usage": cpu_usage,                 
        "memory_usage": memory_usage_pct,       
        "heap_total_bytes": total_mem,
        "heap_free_bytes": free_mem,
        "heap_used_bytes": heap_used_bytes,
        "heap_used_ratio": heap_used_ratio,
        "heap_growth_bps": deltas["heap_growth_bps"],
        "rest_ok": rest_ok,
        "rest_rtt_ms": rest_rtt_ms,
        "pid": pid,
        "proc_cpu_pct": proc["proc_cpu_pct"],
        "proc_rss_kb": proc["proc_rss_kb"],
        "threads": proc["threads"],
        "fd_count": proc["fd_count"],
        "ygc": gc.get("ygc", 0.0),
        "fgc": gc.get("fgc", 0.0),
        "ygct": gc.get("ygct", 0.0),
        "fgct": gc.get("fgct", 0.0),
        "ygc_per_s": deltas["ygc_per_s"],
        "fgc_per_s": deltas["fgc_per_s"],
        "gc_overhead_ratio": deltas["gc_overhead_ratio"],
    }

def collect_all_controllers():
    """Tüm controllerları dön."""
    results = {}
    for ctrl in CONTROLLERS:
        results[ctrl["name"]] = collect_metrics_for(ctrl)
    return results