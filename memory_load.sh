#!/bin/bash

# Starting memory in GB
START_MEMORY=1.1
# Increment in GB
INCREMENT=0.5
# Timeout for each stress test in seconds
TIMEOUT=10
# Maximum memory to stress test (set your limit here, e.g., 5 GB)
MAX_MEMORY=1.8

current_memory=$START_MEMORY

echo "Starting memory stress test..."
while (( $(echo "$current_memory <= $MAX_MEMORY" | bc -l) )); do
    memory_in_mb=$(echo "$current_memory * 1024" | bc)
    echo "Stressing memory: ${memory_in_mb} MB for ${TIMEOUT} seconds..."
    stress --vm 1 --vm-bytes "${memory_in_mb}M" --timeout $TIMEOUT
    current_memory=$(echo "$current_memory + $INCREMENT" | bc)
done

echo "Memory stress test completed."
