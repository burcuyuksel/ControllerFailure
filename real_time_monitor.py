import time
import json
import subprocess
from collect_metrics import collect_metrics
from preprocess_real_time import preprocess_real_time
from predict_real_time import predict_real_time

# Configuration - Floodlight REST API
controller_url = "192.168.56.107:8080"  # Floodlight controller
log_file = "real_time_metrics.log"
sequence_length = 10
mininet_vm_ip = "192.168.56.101"  # Replace with Mininet VM's IP
mininet_user = "mininet"             # Replace with Mininet VM's username
# Switch-to-Controller Mapping
switch_controller_map = {
    "s1": "tcp:192.168.56.117:6653",
    "s2": "tcp:192.168.56.117:6653",
    "s3": "tcp:192.168.56.117:6653"
}
# Track predictions
consecutive_predictions = 0  # Counter for consecutive predictions of 1
# Real-time monitoring loop
while True:
    # Collect metrics
    metrics = collect_metrics(controller_url)
    if metrics:
        with open(log_file, "a") as log:
            log.write(json.dumps(metrics) + "\n")
        
        # Preprocess and predict if enough data is collected
        X_real_time = preprocess_real_time(log_file, sequence_length=sequence_length)
        if len(X_real_time) > 0:
            predictions = predict_real_time(X_real_time)
            latest_prediction = predictions[-1]
            print(f"Latest Prediction: {latest_prediction}")
            # Take action based on prediction
            # Check for consecutive predictions of 1
            if latest_prediction == 1:
                consecutive_predictions += 1
                if consecutive_predictions >= 3:
                    print("3 consecutive predictions of 1 detected. Initiating switch change...")
                    for switch_name, new_controller in switch_controller_map.items():
                        command = f"sudo ovs-vsctl set-controller {switch_name} {new_controller}"
                        ssh_command = f"ssh {mininet_user}@{mininet_vm_ip} '{command}'"
                        try:
                            result = subprocess.run(ssh_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            print("Failover command executed successfully:", result.stdout.decode())
                        except subprocess.CalledProcessError as e:
                            print("Error while executing failover command:", e.stderr.decode())
                        consecutive_predictions = 0  # Reset the counter after action
            else:
                consecutive_predictions = 0  # Reset counter if no failure is predicted
            if latest_prediction == 2:
                print("Failure detected! Initiating failover...")
                # Add failover logic here
    time.sleep(0.5)  # Adjust interval as needed
