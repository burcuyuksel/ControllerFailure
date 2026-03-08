#!/bin/bash
# =============================================================
# Floodlight JVM Stress Script
# Floodlight'ın JVM heap'ini ve CPU'sunu kademeli artırır.
# Doğrudan hedef controller VM'inde çalıştır.
# Kullanım: ./stress_controller.sh [rest_port] [süre] [kademe]
# Örnek:    ./stress_controller.sh 8080 90 3
# =============================================================

REST_PORT="${1:-8080}"
TOTAL_DURATION="${2:-90}"
STAGES="${3:-3}"
STAGE_DURATION=$((TOTAL_DURATION / STAGES))

# Ağır endpointler (JVM'de obje oluşturur → heap dolar)
ENDPOINTS=(
    "/wm/core/controller/switches/json"
    "/wm/core/memory/json"
    "/wm/core/switch/all/flow/json"
    "/wm/core/switch/all/port/json"
    "/wm/core/switch/all/desc/json"
    "/wm/statistics/bandwidth/all/all/json"
)

echo "================================================="
echo " Floodlight JVM Stress (port ${REST_PORT})"
echo " Süre: ${TOTAL_DURATION}s | Kademe: ${STAGES}"
echo " Durdurmak için: Ctrl+C"
echo "================================================="

cleanup() {
    echo ""
    echo "[STOP] Durduruluyor..."
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
    echo "[DONE]"
    exit 0
}
trap cleanup SIGINT SIGTERM

flood_worker() {
    local end_time=$((SECONDS + $1))
    while [ $SECONDS -lt $end_time ]; do
        for ep in "${ENDPOINTS[@]}"; do
            curl -s "http://127.0.0.1:${REST_PORT}${ep}" > /dev/null 2>&1
        done
    done
}

for stage in $(seq 1 $STAGES); do
    workers=$((stage * 5))  # 5, 10, 15 paralel worker
    remaining=$((TOTAL_DURATION - (stage - 1) * STAGE_DURATION))

    echo ""
    echo "[KADEME ${stage}/${STAGES}] ${workers} paralel worker | ${STAGE_DURATION}s"

    for i in $(seq 1 $workers); do
        flood_worker $remaining &
    done

    sleep $STAGE_DURATION
done

wait
cleanup
