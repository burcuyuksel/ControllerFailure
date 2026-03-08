#!/bin/bash
# =============================================================
# Floodlight Controller Stress → Fail Script
# Önce CPU'yu kademeli artırır, sonra controller'ı öldürür.
# Kullanım: ./stress_and_kill.sh [rest_port] [stres_süresi] [worker_sayısı]
# Örnek:    ./stress_and_kill.sh 8081 60 10
# =============================================================

CONTROLLER_IP="192.168.56.107"
REST_PORT="${1:-8080}"
STRESS_DURATION="${2:-60}"    # Stres süresi (saniye)
WORKERS="${3:-10}"

ENDPOINTS=(
    "/wm/core/controller/switches/json"
    "/wm/core/memory/json"
    "/wm/core/switch/all/flow/json"
)

echo "================================================="
echo " FAZA 1: STRES (${STRESS_DURATION} saniye)"
echo " Hedef: ${CONTROLLER_IP}:${REST_PORT}"
echo " Worker: ${WORKERS}"
echo "================================================="

# Worker fonksiyonu
flood_worker() {
    local end_time=$((SECONDS + $1))
    while [ $SECONDS -lt $end_time ]; do
        for ep in "${ENDPOINTS[@]}"; do
            curl -s "http://${CONTROLLER_IP}:${REST_PORT}${ep}" > /dev/null 2>&1
        done
    done
}

# Kademeli artış: her 10 saniyede daha fazla worker ekle
active_pids=()
batch=$((WORKERS / 3))
if [ $batch -lt 1 ]; then batch=1; fi

echo "[STRES] ${batch} worker başlatılıyor (kademe 1/3)..."
for i in $(seq 1 $batch); do
    flood_worker $STRESS_DURATION &
    active_pids+=($!)
done
sleep $((STRESS_DURATION / 3))

echo "[STRES] +${batch} worker ekleniyor (kademe 2/3)..."
remaining=$((STRESS_DURATION * 2 / 3))
for i in $(seq 1 $batch); do
    flood_worker $remaining &
    active_pids+=($!)
done
sleep $((STRESS_DURATION / 3))

echo "[STRES] +${batch} worker ekleniyor (kademe 3/3 - FULL LOAD)..."
remaining2=$((STRESS_DURATION / 3))
for i in $(seq 1 $batch); do
    flood_worker $remaining2 &
    active_pids+=($!)
done
sleep $((STRESS_DURATION / 3))

# Tüm worker'ları durdur
echo "[STRES] Worker'lar durduruluyor..."
for pid in "${active_pids[@]}"; do
    kill $pid 2>/dev/null
done
wait 2>/dev/null

echo ""
echo "================================================="
echo " FAZA 2: KILL"
echo "================================================="

# OpenFlow port eşleşmesi
declare -A OF_PORTS
OF_PORTS[8080]=6653
OF_PORTS[8081]=7753
OF_PORTS[8082]=8853

OF_PORT=${OF_PORTS[$REST_PORT]}

# Floodlight PID'ini bul (REST portuna göre)
FL_PID=$(lsof -ti :${REST_PORT} -sTCP:LISTEN 2>/dev/null | head -1)

if [ -z "$FL_PID" ]; then
    # OpenFlow portuyla dene
    FL_PID=$(lsof -ti :${OF_PORT} -sTCP:LISTEN 2>/dev/null | head -1)
fi

if [ -n "$FL_PID" ]; then
    echo "[KILL] Floodlight PID=${FL_PID} öldürülüyor..."
    sudo kill -9 $FL_PID
    echo "[KILL] Controller ÖLDÜRÜLDÜ! ✗"
else
    echo "[HATA] Port ${REST_PORT}/${OF_PORT}'de Floodlight süreci bulunamadı!"
fi

echo ""
echo "================================================="
echo " TAMAMLANDI"
echo " Stres: ${STRESS_DURATION} saniye | Sonuç: FAIL"
echo "================================================="
