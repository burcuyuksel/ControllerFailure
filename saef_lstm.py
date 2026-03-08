"""
saef_lstm.py
─────────────────────────────────────────────────────────────────────────────
SAEF Proaktif Orkestratör — LSTM Tabanlı Hata Algılama Entegrasyonu (Optimize Edilmiş)

Güncelleme: Just-in-Time (Tam Zamanında) Migration
- P(FAIL) eşiği %35'e çekildi (Erken paniği önlemek için).
- P(FAIL) slope (eğim) hassasiyeti 0.03'e çıkarıldı.
- Stres testinin tamamlanmasına izin veren, sadece çöküşten hemen önceki 2-4 saniyeyi hedefleyen yapı.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import random
import time
import logging
import threading

import numpy as np
import joblib
import pandas as pd
import requests

# ── Mevcut LSTM pipeline modülleri (değiştirilmez) ────────────────────────
from collect_metrics import collect_metrics_for, CONTROLLERS as CTRL_DICTS
from preprocess_real_time import preprocess_real_time
from predict_real_time import predict_real_time
from shap_explain import build_explainer, explain_prediction

# ── Yapılandırma ───────────────────────────────────────────────────────────
CONTROLLERS = [f"{c['ip']}:{c['of_port']}" for c in CTRL_DICTS]
REST_PORTS = [c["rest_port"] for c in CTRL_DICTS]
SWITCHES = ["00:00:00:00:00:00:00:01"]

SW_NAMES = {"00:00:00:00:00:00:00:01": "s1"}
CP_NAMES  = {CONTROLLERS[i]: CTRL_DICTS[i]["name"] for i in range(len(CTRL_DICTS))}

# --- Hassas Ayar Parametreleri ---
P_FAIL_HISTORY_LEN     = 5
STRESS_CONF_THRESHOLD  = 0.90  # Model stresden çok emin olmalı
MIGRATION_FAIL_THRESHOLD = 0.35 # P(Fail) bu sınırı geçmeden göç başlamaz
MIGRATION_SLOPE_THRESHOLD = 0.03 # Artış hızı (ivme) bu sınırı geçmeli
MAX_HEAP_THRESHOLD_MB  = 420    # Kritik bellek sınırı (Just-in-time için yukarı çekildi)

NORMAL_BASELINE    = 5
MONITOR_INTERVAL   = 1.0
SEQUENCE_LENGTH    = 3
WARM_UP_SECONDS    = 15

_SHAP_FEATURES = ['cpu_usage', 'memory_usage', 'heap_mb',
                  'heap_growth_bps', 'proc_cpu_pct', 'rest_rtt_ms']

def _log_path(ctrl_name: str) -> str:
    return f"lstm_{ctrl_name}.log"

# ── Loglama ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SAEF_LSTM")
logging.getLogger("shap").setLevel(logging.WARNING)

class SAEF_Orchestrator:

    def __init__(self):
        self.active_controllers = list(CONTROLLERS)
        self.plan_bank: dict     = {}
        self.load_data           = {cp: 0.1 for cp in CONTROLLERS}
        self.alpha               = 0.5
        self.beta                = 0.5
        self.lbm_history: list   = []
        self.lock                = threading.Lock()
        self.start_time          = time.time()
        self.p_fail_history: dict = {cp: [] for cp in CONTROLLERS}
        self.failed_controllers: set = set()
        self.normal_seen: dict = {cp: 0 for cp in CONTROLLERS}
        self.controller_down_time: dict = {cp: None for cp in CONTROLLERS}
        self.last_rest_ok_time: dict = {cp: None for cp in CONTROLLERS}

        for ctrl in CTRL_DICTS:
            log_path = _log_path(ctrl["name"])
            open(log_path, "w").close()
            logger.info(f"[INIT] {log_path} sıfırlandı.")

        threading.Thread(target=self._init_shap, daemon=True, name="SHAPInit").start()

    def _init_shap(self):
        try:
            scaler   = joblib.load("scaler.pkl")
            df_train = pd.read_csv("dataset_7mart.csv")
            X_all    = scaler.transform(df_train[_SHAP_FEATURES].values)
            bg = np.array([X_all[i:i + SEQUENCE_LENGTH]
                           for i in range(0, min(600, len(X_all) - SEQUENCE_LENGTH), 10)])
            build_explainer(bg)
            logger.info(f"[SHAP] Explainer hazır.")
        except Exception as e:
            logger.warning(f"[SHAP] Explainer kurulamadı: {e}")

    def get_sw_name(self, dpid: str) -> str:
        return SW_NAMES.get(dpid, dpid)

    def get_cp_name(self, cp: str) -> str:
        return CP_NAMES.get(cp, cp)

    def get_latency(self, sw_id: str, target_cp: str) -> float:
        try:
            idx  = CONTROLLERS.index(target_cp)
            port = REST_PORTS[idx]
            url  = f"http://192.168.56.107:{port}/wm/core/switch/{sw_id}/desc/json"
            t0   = time.time()
            resp = requests.get(url, timeout=0.5)
            if resp.status_code == 200:
                return max((time.time() - t0) * 500.0, 0.1)
            return 100.0
        except Exception:
            return 200.0

    def calculate_avg_processing_time(self, cp: str) -> float:
        return 1.0 + (self.load_data[cp] * 10.0)

    def calculate_avg_propagation_delay(self) -> float:
        total, count = 0.0, 0
        for sw in SWITCHES:
            for cp in self.active_controllers:
                total += self.get_latency(sw, cp)
                count += 1
        return total / count if count > 0 else 5.0

    def calculate_lbm(self):
        loads = [self.load_data[cp] for cp in self.active_controllers]
        if not loads: return 0.0, 0.0
        avg = sum(loads) / len(loads)
        curr = max(loads) / avg if avg > 0 else 1.0
        self.lbm_history.append(curr)
        return curr, sum(self.lbm_history) / len(self.lbm_history)

    def adjust_weights_proactively(self):
        d_avg  = self.calculate_avg_propagation_delay()
        t_proc = (sum(self.calculate_avg_processing_time(cp) for cp in self.active_controllers) / max(len(self.active_controllers), 1))
        ratio  = d_avg / t_proc if t_proc > 0 else 1.0
        if ratio >= 2: self.alpha = 1.0
        elif ratio > 0: self.alpha = ratio * 0.5
        self.beta = 1.0 - self.alpha
        logger.info(f"[DYN WEIGHTS] D_avg={d_avg:.2f} T_proc={t_proc:.2f} ratio={ratio:.2f} → α={self.alpha:.2f} β={self.beta:.2f}")

    def cost_function(self, plan: dict, active_list: list) -> float:
        total_lat = sum(self.get_latency(sw, cp) for sw, cp in plan.items())
        norm_delay = min((total_lat / max(len(plan), 1)) / 20.0, 1.0)
        sim_loads  = {cp: 0 for cp in active_list}
        for cp in plan.values(): sim_loads[cp] += 1
        loads = list(sim_loads.values())
        mean  = sum(loads) / len(loads)
        lsd   = math.sqrt(sum((x - mean) ** 2 for x in loads) / len(loads))
        return (self.alpha * norm_delay) + (self.beta * (lsd / 3.0))

    def run_sa_for_scenario(self, scenario_name: str, active_list: list) -> dict:
        T_e = 100.0
        current_plan = {sw: random.choice(active_list) for sw in SWITCHES}
        current_cost = self.cost_function(current_plan, active_list)
        for _ in range(50):
            neighbor = current_plan.copy()
            sw = SWITCHES[0]
            candidates = [c for c in active_list if c != current_plan[sw]]
            if candidates: neighbor[sw] = random.choice(candidates)
            n_cost = self.cost_function(neighbor, active_list)
            delta = n_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T_e, 1e-9)):
                current_plan, current_cost = neighbor, n_cost
            T_e *= 0.95
        return current_plan

    def log_system_status(self):
        curr_lbm, avg_lbm = self.calculate_lbm()
        logger.info("=== [SİSTEM DURUMU] ===")
        logger.info(f"| Aktif Denetleyiciler: {len(self.active_controllers)}")
        for cp in self.active_controllers:
            logger.info(f"|-- {self.get_cp_name(cp)} Yük: {self.load_data[cp]:.4f}")
        logger.info(f"| LBM: {curr_lbm:.3f}  Ort: {avg_lbm:.3f} | α={self.alpha:.2f} β={self.beta:.2f}")

    def proactive_loop(self):
        while True:
            self.log_system_status()
            self.adjust_weights_proactively()
            with self.lock:
                snapshot = list(self.active_controllers)
            new_plans = {}
            new_plans["NORMAL"] = self.run_sa_for_scenario("NORMAL", snapshot)
            if len(snapshot) > 1:
                for cp in snapshot:
                    hypo = [c for c in snapshot if c != cp]
                    new_plans[f"FAIL_{cp}"] = self.run_sa_for_scenario(f"{self.get_cp_name(cp)} ÇÖKERSE", hypo)
            with self.lock:
                self.plan_bank = new_plans
            time.sleep(3)

    def _run_lstm_for(self, cp_addr: str, ctrl_dict: dict):
        cp_name  = ctrl_dict["name"]
        log_file = _log_path(cp_name)
        metrics = collect_metrics_for(ctrl_dict)

        if metrics:
            if metrics.get("rest_ok", 1) == 0:
                if self.controller_down_time[cp_addr] is None:
                    true_fail_t0 = self.last_rest_ok_time[cp_addr] or time.time()
                    self.controller_down_time[cp_addr] = true_fail_t0
                    logger.error(f"[DOWN-DETECT] {cp_name} REST DOWN. Gerçek t0: {true_fail_t0:.3f}")
            else:
                self.last_rest_ok_time[cp_addr] = time.time()
                self.controller_down_time[cp_addr] = None

            with open(log_file, "a") as fh:
                fh.write(json.dumps({"c1": metrics}) + "\n")

            rtt = metrics.get("rest_rtt_ms", 0.0) or 0.0
            self.load_data[cp_addr] = min(rtt / 1500.0, 1.0)

            try:
                X = preprocess_real_time(log_file, sequence_length=SEQUENCE_LENGTH)
                if len(X) == 0: return
                preds = predict_real_time(X)
                if not preds: return

                pred, stress_prob, fail_prob = preds[0]
                label = ["NORMAL", "STRESS", "FAIL"][pred]

                if pred in (1, 2):
                    try: explain_prediction(X[-1:], pred)
                    except: pass

                if pred == 0:
                    self.normal_seen[cp_addr] = min(self.normal_seen[cp_addr] + 1, NORMAL_BASELINE)

                pf_history = self.p_fail_history[cp_addr]
                pf_history.append(fail_prob)
                if len(pf_history) > P_FAIL_HISTORY_LEN: pf_history.pop(0)

                elapsed = time.time() - self.start_time
                in_warmup = elapsed < WARM_UP_SECONDS
                has_baseline = self.normal_seen[cp_addr] >= NORMAL_BASELINE

                logger.info(f"[LSTM-{cp_name}] Tahmin: {label} P(S)={stress_prob:.2f} P(F)={fail_prob:.2f}")

                # ── JUST-IN-TIME Karar Mantığı ──────────────────────────────────
                trend_ok = False
                slope = 0.0
                if len(pf_history) == P_FAIL_HISTORY_LEN:
                    slope, _ = np.polyfit(range(P_FAIL_HISTORY_LEN), pf_history, 1)
                    # Sınır: %35 olasılık VE dik artış
                    trend_ok = (slope > MIGRATION_SLOPE_THRESHOLD and fail_prob > MIGRATION_FAIL_THRESHOLD)

                heap_mb = metrics.get("heap_mb", 0.0) or 0.0
                proactive_metric_trigger = (stress_prob > STRESS_CONF_THRESHOLD and heap_mb > MAX_HEAP_THRESHOLD_MB)

                if pred == 2 or trend_ok or proactive_metric_trigger:
                    signal_summary = f"P(F)_slope={slope:.3f} P(F)={fail_prob:.2f} Heap={heap_mb:.1f}"

                    if in_warmup or not has_baseline:
                        logger.warning(f"[LSTM-{cp_name}] Tehlike algılandı ama kısıtlar (warmup/baseline) nedeniyle beklendi. [{signal_summary}]")
                        self.p_fail_history[cp_addr].clear()
                    elif cp_addr not in self.failed_controllers:
                        logger.error(f"!!! KRİTİK EŞİK GEÇİLDİ: {cp_name} ÇÖKÜYOR — MİGRASYON BAŞLATILDI !!! [{signal_summary}]")
                        with self.lock:
                            if cp_addr in self.active_controllers: self.active_controllers.remove(cp_addr)
                        self.failed_controllers.add(cp_addr)
                        self.trigger_recovery(cp_addr, time.time())
                        self.p_fail_history[cp_addr].clear()

                elif slope > 0.01:
                    logger.info(f"[LSTM-{cp_name}] Stres altında stabil: slope={slope:.3f}, P(F)={fail_prob:.2f}")

            except Exception as exc:
                logger.warning(f"[LSTM-{cp_name}] Tahmin hatası: {exc}")

    def lstm_monitor_loop(self, cp_addr: str, ctrl_dict: dict):
        while True:
            if cp_addr in self.failed_controllers: break
            self._run_lstm_for(cp_addr, ctrl_dict)
            time.sleep(MONITOR_INTERVAL)

    def trigger_recovery(self, failed_cp: str, start_time: float):
        logger.warning("=== [FAILOVER AKSİYONU BAŞLADI] ===")
        with self.lock:
            recovery_plan = self.plan_bank.get(f"FAIL_{failed_cp}")

        if not recovery_plan:
            with self.lock: remaining = [c for c in self.active_controllers if c != failed_cp]
            if not remaining: return
            recovery_plan = self.run_sa_for_scenario(f"FALLBACK", remaining)

        action_count = 0
        for sw, target_cp in recovery_plan.items():
            try:
                idx, port = CONTROLLERS.index(target_cp), REST_PORTS[CONTROLLERS.index(target_cp)]
                url = f"http://192.168.56.107:{port}/wm/core/switch/{sw}/role/json"
                requests.post(url, json={"role": "MASTER"}, timeout=0.5)
                logger.info(f"|-- [MIGRATION] {self.get_sw_name(sw)} → {self.get_cp_name(target_cp)}")
                action_count += 1
            except: pass

        done_time = time.time()
        t0 = self.controller_down_time.get(failed_cp)

        if not t0:
            logger.warning(f"\n{'='*60}\n [BEKLEME] ✅ MİGRASYON TAMAMLANDI. Gerçek fail bekleniyor...\n{'='*60}")
            idx, port = CONTROLLERS.index(failed_cp), REST_PORTS[CONTROLLERS.index(failed_cp)]
            url = f"http://192.168.56.107:{port}/wm/core/health/json"
            while True:
                try:
                    resp = requests.get(url, timeout=0.5)
                    if resp.status_code != 200: t0 = time.time(); break
                except: t0 = time.time(); break
                time.sleep(0.1)
            logger.warning(f" --> Controller GERÇEKTEN fail oldu (t0={t0:.3f})")

        gain_ms = (t0 - done_time) * 1000
        logger.warning(f"\n{'='*60}\n [ZAMAN ÖLÇÜMÜ] Proaktif Kazanç: {gain_ms:.1f} ms\n{'='*60}")
        import os; os._exit(0)

    def run(self):
        logger.info("SAEF-LSTM Orkestratör Başlatıldı (Just-in-Time Mode).")
        threading.Thread(target=self.proactive_loop, daemon=True).start()
        c1_dict, c1_addr = CTRL_DICTS[0], CONTROLLERS[0]
        t_lstm = threading.Thread(target=self.lstm_monitor_loop, args=(c1_addr, c1_dict), daemon=True)
        t_lstm.start()
        t_lstm.join()

if __name__ == "__main__":
    SAEF_Orchestrator().run()
