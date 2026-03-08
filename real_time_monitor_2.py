import time
import json
import numpy as np
import traceback
import pandas as pd
import joblib
from collect_metrics import collect_metrics_for, CONTROLLERS
from preprocess_real_time import preprocess_real_time
from predict_real_time import predict_real_time
from shap_explain import build_explainer, explain_prediction

# ── Konfigürasyon ─────────────────────────────────────────────
log_file        = "real_time_metrics.log"
sequence_length = 3   # lstm_model.pth ile aynı olmalı
controller_name = CONTROLLERS[0]["name"]

FEATURES = ['cpu_usage', 'memory_usage', 'heap_mb',
            'heap_growth_bps', 'proc_cpu_pct', 'rest_rtt_ms']

# ── SHAP Explainer'ı startup'ta bir kez kur ──────────────────
print("SHAP explainer oluşturuluyor (30-60 sn sürebilir)...")
try:
    scaler    = joblib.load("scaler.pkl")
    df_train  = pd.read_csv("dataset_7mart.csv")
    X_all     = scaler.transform(df_train[FEATURES].values)
    # 10 adımlık sequence'lerden 60 arka plan örneği al
    bg = np.array([X_all[i:i+sequence_length]
                   for i in range(0, min(600, len(X_all) - sequence_length), 10)])
    build_explainer(bg)
    print(f"SHAP explainer hazır ({len(bg)} arka plan örneği).\n")
except Exception as e:
    print(f"[UYARI] SHAP explainer kurulamadı: {e}")

# ── Gerçek Zamanlı İzleme Döngüsü ────────────────────────────
print(f"Monitoring Controller: {controller_name} ({CONTROLLERS[0]['ip']})")
print(f"Watching {log_file}... (ilk {sequence_length} tahmin için veri bekleniyor)\n")

recent_predictions = []

try:
    while True:
        metrics = collect_metrics_for(CONTROLLERS[0])
        if metrics:
            with open(log_file, "a") as log:
                log.write(json.dumps({controller_name: metrics}) + "\n")

            try:
                X_real_time = preprocess_real_time(log_file, sequence_length=sequence_length)

                if len(X_real_time) > 0:
                    predictions = predict_real_time(X_real_time)
                    latest_prediction, stress_prob, fail_prob = predictions[0]

                    recent_predictions.append(latest_prediction)
                    if len(recent_predictions) > 3:
                        recent_predictions.pop(0)

                    labels      = ["NORMAL", "STRESS", "FAIL"]
                    fail_count  = recent_predictions.count(2)
                    stress_count= recent_predictions.count(1)

                    print(f"[{time.strftime('%H:%M:%S')}] Raw Pred: {labels[latest_prediction]} "
                          f"P(S)={stress_prob:.2f} P(F)={fail_prob:.2f} | "
                          f"Window: [N:{recent_predictions.count(0)} S:{stress_count} F:{fail_count}]")

                    if fail_count >= 2:
                        print(">>> FAILURE DETECTED! Initiating failover... <<<")
                        explain_prediction(X_real_time, predicted_label=2)
                        recent_predictions.clear()

                    elif stress_count >= 2:
                        print(">>> STRESS DETECTED! System load is high. <<<")
                        explain_prediction(X_real_time, predicted_label=1)
                        recent_predictions.clear()

            except Exception as e:
                print(f"Prediction Error: {e}")
                traceback.print_exc()

        time.sleep(1.0)

except KeyboardInterrupt:
    print("\nMonitoring stopped by user.")