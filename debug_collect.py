import time
import requests
import subprocess
import os

print("Starting debug run...")

def run_cmd(cmd):
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2.0)
        dur = time.time() - t0
        print(f"[{dur:.3f}s] CMD: {cmd[:40]}... (Code: {proc.returncode})")
    except subprocess.TimeoutExpired:
        dur = time.time() - t0
        print(f"[{dur:.3f}s] TIMEOUT: {cmd[:40]}...")
    except Exception as e:
        print(f"ERROR {cmd}: {e}")

print("1. Testing REST call to c1...")
t0 = time.time()
try:
    r = requests.get("http://192.168.56.107:8080/wm/core/memory/json", timeout=2.0)
    dur = time.time() - t0
    print(f"[{dur:.3f}s] REST: OK")
except Exception as e:
    dur = time.time() - t0
    print(f"[{dur:.3f}s] REST FAILED: {type(e).__name__}")

print("2. Testing System CPU read...")
t0 = time.time()
with open('/proc/stat', 'r') as f:
    line1 = f.readline()
time.sleep(0.3)
with open('/proc/stat', 'r') as f:
    line2 = f.readline()
dur = time.time() - t0
print(f"[{dur:.3f}s] CPU Read")

print("3. Testing Process Commands...")
# Find a floodlight PID dynamically (assume 8080 is c1)
out = subprocess.run("ss -lptn | grep ':8080 '", shell=True, capture_output=True, text=True)
import re
match = re.search(r'pid=(\d+)', out.stdout)
if match:
    pid = match.group(1)
    print(f"Found Floodlight PID: {pid}")
    run_cmd(f"ps -p {pid} -o %cpu=,rss=,nlwp=")
    run_cmd(f"jstat -gc {pid} 1000 1 2>/dev/null")
    
    t0 = time.time()
    try:
        fd_count = len(os.listdir(f"/proc/{pid}/fd"))
        dur = time.time() - t0
        print(f"[{dur:.3f}s] FD Count: {fd_count}")
    except:
        pass
else:
    print("Could not find Floodlight PID on 8080.")

print("Done.")
