"""
saef_lstm.py
─────────────────────────────────────────────────────────────────────────────
SAEF Proaktif Orkestratör — LSTM Tabanlı Hata Algılama Entegrasyonu

Mimari:
  Thread-1 (daemon)  : proactive_loop()
      → Simulated Annealing ile her senaryo için switch→controller atama planı üretir.
      → Her 3 sn'de plan_bank'ı günceller.

  Thread-2..4 (daemon): lstm_monitor_loop(controller)
      → Her controller için ayrı thread çalışır.
      → collect_metrics_for() → log → preprocess → predict_real_time()
      → Sliding-window (son 3 tahmin): fail_count>=2 → trigger_recovery()

  Thread-5 (main join): orchestrator thread, tüm thread'lerin yaşamasını bekler.

Değiştirilmeyen modüller (hiç dokunulmaz):
  collect_metrics.py, preprocess_real_time.py, predict_real_time.py,
  lstm_train.py, shap_explain.py
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
# CTRL_DICTS örneği:
#   [{"name":"c1","ip":"192.168.56.107","rest_port":"8080","of_port":"6653"}, ...]

# saef.py ile tutarlı: host:of_port formatında controller adresi listesi
CONTROLLERS = [f"{c['ip']}:{c['of_port']}" for c in CTRL_DICTS]

# REST portları (plan bankası için latency ölçümünde kullanılır)
REST_PORTS = [c["rest_port"] for c in CTRL_DICTS]

# DPID listesi
SWITCHES = ["00:00:00:00:00:00:00:01"]  # Yalnızca s1

# Kısa isim eşleşmeleri (log okunabilirliği için)
SW_NAMES = {"00:00:00:00:00:00:00:01": "s1"}  # Yalnızca s1
CP_NAMES  = {CONTROLLERS[i]: CTRL_DICTS[i]["name"] for i in range(len(CTRL_DICTS))}

# Sinyal bazlı migration için son 5 P(FAIL) değeri tutulur
P_FAIL_HISTORY_LEN     = 5
STRESS_CONF_THRESHOLD  = 0.75  # P(STRESS) ≥ bu değer olmalı (model emin olmalı)
STRESS_CPU_THRESHOLD   = 85.0  # cpu_usage % — metrik eğimi için eşik

# Failover için gereken minimum "NORMAL" taban çizgisi
# Bu kadar NORMAL tahmin görülmeden FAIL tetiklenmez
NORMAL_BASELINE    = 5

# Metrik toplama sıklığı (sn) — preprocess için sequence_length kadar veri birikmesi gerekir
MONITOR_INTERVAL   = 1.0
SEQUENCE_LENGTH    = 3    # lstm_model.pth ile aynı olmalı

_SHAP_FEATURES = ['cpu_usage', 'memory_usage', 'heap_mb',
                  'heap_growth_bps', 'proc_cpu_pct', 'rest_rtt_ms']

# Her controller için ayrı log dosyası
def _log_path(ctrl_name: str) -> str:
    return f"lstm_{ctrl_name}.log"

# Warm-up süresi (sn): Bu süre boyunca failover tetiklenmez
WARM_UP_SECONDS = 15

# ── Loglama ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SAEF_LSTM")

# SHAP kütüphanesinin iç mesajlarını (phi, kernelWeights ...) sustur
logging.getLogger("shap").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════
# SAEF_Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
class SAEF_Orchestrator:

    def __init__(self):
        self.active_controllers = list(CONTROLLERS)
        self.plan_bank: dict     = {}
        self.load_data           = {cp: 0.1 for cp in CONTROLLERS}
        self.alpha               = 0.5
        self.beta                = 0.5
        self.lbm_history: list   = []
        self.lock                = threading.Lock()
        self.start_time          = time.time()   # warm-up referansı

        # Sinyal bazlı trend analizi için son 5 P(FAIL) değerini tutar
        # {cp_addr: [pf0, pf1, pf2, pf3, pf4]}
        self.p_fail_history: dict = {cp: [] for cp in CONTROLLERS}

        # Failover yapılıp yapılmadığını takip et (çift tetiklemeyi önler)
        self.failed_controllers: set = set()

        # Taban çizgisi sayacı: en az NORMAL_BASELINE kez NORMAL görülmeli
        # {cp_addr: int}  — her NORMAL +1, maksimum NORMAL_BASELINE'da doyar
        self.normal_seen: dict = {cp: 0 for cp in CONTROLLERS}

        # Controller'ın REST'e erişilemez olduğu ilk an (rest_ok==0)
        # {cp_addr: float}  — unix timestamp, bir kez set edilir
        self.controller_down_time: dict = {cp: None for cp in CONTROLLERS}

        # Son başarılı REST yanıtı zamanı (2W-FD'den ilham: gerçek fail t₀ referansı)
        # Her rest_ok==1 geldiğinde güncellenir.
        # Controller fail olduğunda bu, "controller'a son ulaşılan anı" gösterir.
        self.last_rest_ok_time: dict = {cp: None for cp in CONTROLLERS}

        # Startup: her controller log dosyasını sıfırla (eski kirli veriyi temizle)
        for ctrl in CTRL_DICTS:
            log_path = _log_path(ctrl["name"])
            open(log_path, "w").close()   # truncate
            logger.info(f"[INIT] {log_path} sıfırlandı.")

        # SHAP explainer'ı arka planda kur (başlangıç süresini etkilemesin)
        threading.Thread(target=self._init_shap, daemon=True, name="SHAPInit").start()

    # ── SHAP başlatma (arka planda) ─────────────────────────────────────
    def _init_shap(self):
        logger.info("[SHAP] Explainer oluşturuluyor (30-60 sn sürebilir)...")
        try:
            scaler   = joblib.load("scaler.pkl")
            df_train = pd.read_csv("dataset_7mart.csv")
            X_all    = scaler.transform(df_train[_SHAP_FEATURES].values)
            bg = np.array([X_all[i:i + SEQUENCE_LENGTH]
                           for i in range(0, min(600, len(X_all) - SEQUENCE_LENGTH), 10)])
            build_explainer(bg)
            logger.info(f"[SHAP] Explainer hazır ({len(bg)} arka plan örneği).")
        except Exception as e:
            logger.warning(f"[SHAP] Explainer kurulamadı: {e}")

    # ── Yardımcı isim dönüşümleri ─────────────────────────────────────────
    def get_sw_name(self, dpid: str) -> str:
        return SW_NAMES.get(dpid, dpid)

    def get_cp_name(self, cp: str) -> str:
        return CP_NAMES.get(cp, cp)

    # ── Yük ve gecikme metrikleri ─────────────────────────────────────────
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
        if not loads:
            return 0.0, 0.0
        avg   = sum(loads) / len(loads)
        curr  = max(loads) / avg if avg > 0 else 1.0
        self.lbm_history.append(curr)
        return curr, sum(self.lbm_history) / len(self.lbm_history)

    # ── Dinamik ağırlık ayarlama ──────────────────────────────────────────
    def adjust_weights_proactively(self):
        d_avg  = self.calculate_avg_propagation_delay()
        t_proc = (sum(self.calculate_avg_processing_time(cp)
                      for cp in self.active_controllers)
                  / max(len(self.active_controllers), 1))
        ratio  = d_avg / t_proc if t_proc > 0 else 1.0
        if ratio >= 2:
            self.alpha = 1.0
        elif ratio > 0:
            self.alpha = ratio * 0.5
        self.beta = 1.0 - self.alpha
        logger.info(
            f"[DYN WEIGHTS] D_avg={d_avg:.2f} T_proc={t_proc:.2f} "
            f"ratio={ratio:.2f} → α={self.alpha:.2f} β={self.beta:.2f}"
        )

    # ── Maliyet fonksiyonu ─────────────────────────────────────────────────
    def cost_function(self, plan: dict, active_list: list) -> float:
        total_lat = sum(self.get_latency(sw, cp) for sw, cp in plan.items())
        norm_delay = min((total_lat / max(len(plan), 1)) / 20.0, 1.0)
        sim_loads  = {cp: 0 for cp in active_list}
        for cp in plan.values():
            sim_loads[cp] += 1
        loads = list(sim_loads.values())
        mean  = sum(loads) / len(loads)
        lsd   = math.sqrt(sum((x - mean) ** 2 for x in loads) / len(loads))
        return (self.alpha * norm_delay) + (self.beta * (lsd / 3.0))

    # ── Simulated Annealing ───────────────────────────────────────────────
    def run_sa_for_scenario(self, scenario_name: str, active_list: list) -> dict:
        logger.info(f"[{scenario_name} - SA BAŞLADI]")
        T_e          = 100.0
        current_plan = {sw: random.choice(active_list) for sw in SWITCHES}
        current_cost = self.cost_function(current_plan, active_list)
        init_cost    = current_cost

        for _ in range(50):
            neighbor = current_plan.copy()
            # Tek switch: s1'e atanan controller'ı rastgele değiştir
            sw = SWITCHES[0]
            candidates = [c for c in active_list if c != current_plan[sw]]
            if candidates:
                neighbor[sw] = random.choice(candidates)
            n_cost = self.cost_function(neighbor, active_list)
            delta  = n_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T_e, 1e-9)):
                current_plan, current_cost = neighbor, n_cost
            T_e *= 0.95

        pretty = {self.get_sw_name(sw): self.get_cp_name(cp)
                  for sw, cp in current_plan.items()}
        logger.info(
            f"[{scenario_name} - SA BİTTİ] "
            f"maliyet: {init_cost:.4f} → {current_cost:.4f} | atama: {pretty}"
        )
        return current_plan

    # ── Sistem durumu logu ────────────────────────────────────────────────
    def log_system_status(self):
        curr_lbm, avg_lbm = self.calculate_lbm()
        logger.info("=== [SİSTEM DURUMU] ===")
        logger.info(f"| Aktif Denetleyiciler: {len(self.active_controllers)}")
        for cp in self.active_controllers:
            logger.info(f"|-- {self.get_cp_name(cp)} Yük: {self.load_data[cp]:.4f}")
        logger.info(f"| LBM: {curr_lbm:.3f}  Ort: {avg_lbm:.3f}")
        logger.info(f"| α={self.alpha:.2f}  β={self.beta:.2f}")
        logger.info("=======================")

    # ══════════════════════════════════════════════════════════════════════
    # Thread-1: Proaktif plan bankası güncelleme (SA)
    # ══════════════════════════════════════════════════════════════════════
    def proactive_loop(self):
        while True:
            self.log_system_status()
            self.adjust_weights_proactively()
            logger.info("[PLAN BANKASI GÜNCELLEME]")

            # Döngü boyunca tutarlı olması için snapshot al
            with self.lock:
                snapshot = list(self.active_controllers)

            new_plans = {}
            new_plans["NORMAL"] = self.run_sa_for_scenario("NORMAL", snapshot)

            # Her controller için FAIL senaryosu planla (saef.py ile tutarlı)
            if len(snapshot) > 1:
                for cp in snapshot:
                    hypo = [c for c in snapshot if c != cp]
                    new_plans[f"FAIL_{cp}"] = self.run_sa_for_scenario(
                        f"{self.get_cp_name(cp)} ÇÖKERSE", hypo
                    )

            with self.lock:
                self.plan_bank = new_plans
            time.sleep(3)

    # ══════════════════════════════════════════════════════════════════════
    # Thread-2..N: LSTM izleme döngüsü — her controller için ayrı thread
    # ══════════════════════════════════════════════════════════════════════
    def _run_lstm_for(self, cp_addr: str, ctrl_dict: dict):
        """
        Tek bir controller için:
          1. Metrik topla (collect_metrics_for)
          2. Log dosyasına yaz
          3. Preprocess → predict
          4. Sliding-window kararı ver
        """
        cp_name  = ctrl_dict["name"]
        log_file = _log_path(cp_name)

        # 1. Metrik topla
        metrics = collect_metrics_for(ctrl_dict)

        if metrics:
            # ── REST durumu: gerçek fail zamanını izöle ────────────────────
            if metrics.get("rest_ok", 1) == 0:
                if self.controller_down_time[cp_addr] is None:
                    # Gerçek fail t₀: son başarılı REST andan beri REST yanıtı yok
                    true_fail_t0 = self.last_rest_ok_time[cp_addr] or time.time()
                    self.controller_down_time[cp_addr] = true_fail_t0
                    logger.error(
                        f"[DOWN-DETECT] {cp_name} REST erişilemez — "
                        f"Gerçek fail t₀ (son REST): {true_fail_t0:.3f} "
                        f"| Algılama gecikme: {(time.time() - true_fail_t0)*1000:.1f} ms"
                    )
            else:
                # REST sağlıklı → son başarılı REST anını güncelle
                self.last_rest_ok_time[cp_addr] = time.time()
                self.controller_down_time[cp_addr] = None  # toparlandıysa sıfırla

            # 2. Log dosyasına yaz (preprocess_real_time bunu okur)
            # Önemli: preprocess_real_time.py içinde CONTROLLER_DEFAULT="c1"
            # hardcoded olduğundan, tüm controller logları "c1" anahtarıyla
            # yazılmalıdır. Log dosyasının adı zaten hangi controller olduğunu belirtiyor.
            with open(log_file, "a") as fh:
                fh.write(json.dumps({"c1": metrics}) + "\n")

            # Yük bilgisini güncelle (maliyet fonksiyonu için)
            # bandwidth yerine rest_rtt_ms'i normalleştirerek yük tahmini yap
            rtt = metrics.get("rest_rtt_ms", 0.0) or 0.0
            self.load_data[cp_addr] = min(rtt / 1500.0, 1.0)

            # 3. Preprocess + Predict
            try:
                X = preprocess_real_time(log_file, sequence_length=SEQUENCE_LENGTH)
                if len(X) == 0:
                    logger.info(
                        f"[LSTM-{cp_name}] Henüz yeterli veri yok "
                        f"(en az {SEQUENCE_LENGTH} adım gerekli)"
                    )
                    return

                preds = predict_real_time(X)
                if not preds:
                    return

                pred, stress_prob, fail_prob = preds[0]
                label = ["NORMAL", "STRESS", "FAIL"][pred]

                # XAI: STRESS veya FAIL tespitinde hangi feature'ların
                # kararı ne kadar etkilediğini açıkla
                if pred in (1, 2):  # STRESS veya FAIL
                    try:
                        explain_prediction(X[-1:], pred)  # Son sequence
                    except Exception as xai_exc:
                        logger.debug(f"[XAI-{cp_name}] Açıklama atlandı: {xai_exc}")

                # 4a. Taban çizgisi güncellemesi
                if pred == 0:  # NORMAL → toparlandı
                    self.normal_seen[cp_addr] = min(
                        self.normal_seen[cp_addr] + 1, NORMAL_BASELINE
                    )
                # FAIL veya STRESS geldiğinde normal_seen sıfırlanmaz:
                # Sistem bir kez sağlıklı göründüğünde bu kazanım korunur.

                has_baseline = self.normal_seen[cp_addr] >= NORMAL_BASELINE
                baseline_tag = (
                    f" [baseline: {self.normal_seen[cp_addr]}/{NORMAL_BASELINE}]"
                    if not has_baseline else ""
                )

                # 4b. P(FAIL) geçmişini güncelle
                pf_history = self.p_fail_history[cp_addr]
                pf_history.append(fail_prob)
                if len(pf_history) > P_FAIL_HISTORY_LEN:
                    pf_history.pop(0)

                # Warm-up süresinde mi?
                elapsed = time.time() - self.start_time
                in_warmup = elapsed < WARM_UP_SECONDS
                warmup_tag = f" [WARM-UP {WARM_UP_SECONDS - elapsed:.0f}s]" if in_warmup else ""

                logger.info(
                    f"[LSTM-{cp_name}] Tahmin: {label} "
                    f"P(S)={stress_prob:.2f} P(F)={fail_prob:.2f}"
                    f"{warmup_tag}{baseline_tag}"
                )

                # ── Sinyal Bazlı Migration Kararı (Orijinal Mantık) ─────────────────
                # Üç sinyal aynı anda doğrulanmalı:
                #   1. P(FAIL) Trendi: Son 5 ölçümde net bir artış varsa ve fail_prob > %25
                #   2. Güven: P(STRESS) veya P(FAIL) yüksek olmalı
                
                # np.polyfit ile doğrusal eğimi (slope) hesapla
                trend_ok = False
                slope = 0.0
                if len(pf_history) == P_FAIL_HISTORY_LEN:
                    slope, _ = np.polyfit(range(P_FAIL_HISTORY_LEN), pf_history, 1)
                    # Eğtim pozitifse ve anlık fail oranı > %25 ise (hemen çöküş öncesi)
                    trend_ok = (slope > 0.02 and fail_prob > 0.25)
                
                # Model P(Fail)'i gecikmeli artırdığı için (0.13'te takılıp aniden 0.70'e uçuyor),
                # Alternatif Proaktif Tetikleyici: Model "STRESS" durumundan çok emin ve
                # Heap 350MB sınırını geçtiyse çöküşe 1-2 saniye kalmış demektir.
                heap_mb = metrics.get("heap_mb", 0.0) or 0.0
                proactive_metric_trigger = (stress_prob > 0.80 and heap_mb > 350)
                
                # Ya doğrudan fail (pred==2) durumunda ya da (kuvvetli trend veya proaktif metrik) varsa tetikle
                if pred == 2 or trend_ok or proactive_metric_trigger:
                    if proactive_metric_trigger:
                        signal_summary = f"Proaktif Metrik (P(S)={stress_prob:.2f}, Heap={heap_mb:.1f}MB)"
                    else:
                        signal_summary = f"P(F)_slope={slope:.3f} P(F)={fail_prob:.2f}"
                    if in_warmup:
                        logger.warning(
                            f"[LSTM-{cp_name}] Çok-sinyalli tehlike algılandı ama "
                            f"warm-up süresi ({WARM_UP_SECONDS}s) dolmadı, yoksayıldı. "
                            f"[{signal_summary}]"
                        )
                        self.p_fail_history[cp_addr].clear()
                    elif not has_baseline:
                        logger.warning(
                            f"[LSTM-{cp_name}] Çok-sinyalli tehlike ama "
                            f"taban çizgisi oluşmadı "
                            f"({self.normal_seen[cp_addr]}/{NORMAL_BASELINE} NORMAL). "
                            f"[{signal_summary}]"
                        )
                        self.p_fail_history[cp_addr].clear()
                    elif cp_addr not in self.failed_controllers:
                        logger.error(
                            f"!!! ÇOK-SİNYAL KARARI: {cp_name} TOPARLANAMIYOR — "
                            f"SWITCH MİGRATİON BAŞLATILIYOR !!! "
                            f"[{signal_summary}]"
                        )
                        with self.lock:
                            if cp_addr in self.active_controllers:
                                self.active_controllers.remove(cp_addr)
                        self.failed_controllers.add(cp_addr)
                        detection_ref = time.time()
                        self.trigger_recovery(cp_addr, detection_ref)
                        self.p_fail_history[cp_addr].clear()
                elif slope > 0.01:
                    # Trend kötü ama eşiğe (0.25) henüz gelmedi
                    logger.warning(
                        f"[LSTM-{cp_name}] P(FAIL) artış trendinde (slope={slope:.3f}) ama "
                        f"beklemeye devam: P(F)={fail_prob:.2f}"
                    )

            except Exception as exc:
                logger.warning(f"[LSTM-{cp_name}] Tahmin hatası: {exc}")

    def lstm_monitor_loop(self, cp_addr: str, ctrl_dict: dict):
        """Her controller için sürekli çalışan izleme döngüsü."""
        cp_name = ctrl_dict["name"]
        logger.info(f"[LSTM-{cp_name}] İzleme başladı → {cp_addr}")
        while True:
            # Controller hâlâ aktif mi? Çöktüyse döngüyü durdur.
            if cp_addr in self.failed_controllers:
                logger.info(f"[LSTM-{cp_name}] Denetleyici arızalı, thread sonlanıyor.")
                break
            self._run_lstm_for(cp_addr, ctrl_dict)
            time.sleep(MONITOR_INTERVAL)

    # ══════════════════════════════════════════════════════════════════════
    # Failover: Plan bankasındaki hazır planı uygula
    # ══════════════════════════════════════════════════════════════════════
    def trigger_recovery(self, failed_cp: str, start_time: float):
        logger.warning("=== [FAILOVER AKSİYONU BAŞLADI] ===")
        with self.lock:
            recovery_plan = self.plan_bank.get(f"FAIL_{failed_cp}")

        if not recovery_plan:
            # Plan bankasında hazır plan yok (SA henüz hesaplamamış olabilir).
            # Geriye kalan aktif controller'larla anlık yedek plan hesapla.
            with self.lock:
                remaining = [c for c in self.active_controllers if c != failed_cp]
            if not remaining:
                logger.error(
                    f"[FAILOVER] Hiç aktif controller kalmadı! "
                    f"Migration yapılamıyor."
                )
                return
            logger.warning(
                f"[FAILOVER] '{self.get_cp_name(failed_cp)}' için "
                f"plan bankasında hazır plan yok — anlık yedek plan hesaplanıyor..."
            )
            recovery_plan = self.run_sa_for_scenario(
                f"FALLBACK_{self.get_cp_name(failed_cp)}", remaining
            )

        action_count = 0
        for sw, target_cp in recovery_plan.items():
            try:
                idx      = CONTROLLERS.index(target_cp)
                port     = REST_PORTS[idx]
                url      = (f"http://192.168.56.107:{port}"
                            f"/wm/core/switch/{sw}/role/json")
                requests.post(url, json={"role": "MASTER"}, timeout=0.5)
                logger.info(
                    f"|-- [MIGRATION] "
                    f"{self.get_sw_name(sw)} → {self.get_cp_name(target_cp)}"
                )
                action_count += 1
            except Exception:
                pass

        recovery_done_time = time.time()   # t₂: migration tamamlandı
        # start_time = t₁: LSTM migration kararı anı
        migration_ms = (recovery_done_time - start_time) * 1000

        # ── Üç zaman damgasıyla tam failover zamanölçümü ────────────────────
        # t₀ = controller'a son başarılı ulaşılan an (gerçek fail başlangıcı)
        # t₁ = LSTM migration kararı (detection_ref geçirilen değer)
        # t₂ = migration tamamlama
        t0 = self.controller_down_time.get(failed_cp)  # gerçek fail t₀
        
        if not t0:
            logger.warning(
                f"\n{'='*60}"
                f"\n  [BEKLEME] ✅ PROAKTİF MİGRATİON TAMAMLANDI"
                f"\n  Controller ({self.get_cp_name(failed_cp)}) henüz fail olmadı."
                f"\n  Gerçek fail anı bekleniyor..."
                f"\n{'='*60}"
            )
            idx = CONTROLLERS.index(failed_cp)
            port = REST_PORTS[idx]
            url = f"http://192.168.56.107:{port}/wm/core/health/json"
            
            while True:
                try:
                    resp = requests.get(url, timeout=0.5)
                    if resp.status_code != 200:
                        t0 = time.time()
                        break
                except Exception:
                    t0 = time.time()
                    break
                time.sleep(0.5)
                
            self.controller_down_time[failed_cp] = t0
            logger.warning(f"  --> Controller GERÇEKTEN fail oldu (t₀={t0:.3f})")

        detection_delay_ms = (start_time - t0) * 1000   # t₁ - t₀
        total_ms           = (recovery_done_time - t0) * 1000  # t₂ - t₀
        
        time_gained_ms = (t0 - recovery_done_time) * 1000
        
        if recovery_done_time < t0:
            proaktif_tag = "✅ PROAKTİF (fail olmadan önce migration tamamlandı)"
            gain_line = f"\n  ★ Göçün Fail'den Ne Kadar Önce Bittiği: {time_gained_ms:.1f} ms"
        else:
            proaktif_tag = "⚠️  REAKTİF (fail olduktan sonra algılandı)"
            gain_line = f"\n  ★ Gerçek Fail → Migration Bitti: {total_ms:.1f} ms"

        logger.warning(
            f"\n{'='*60}"
            f"\n  [FAILOVER ZAMAN ÖLÇÜMÜ]  {proaktif_tag}"
            f"\n  t₀  Gerçek Fail (REST Down): {t0:.3f}"
            f"\n  t₁  LSTM Migration Kararı:   {start_time:.3f}"
            f"\n  t₂  Migration Tamamlandı:    {recovery_done_time:.3f}"
            f"\n  ───────────────────────────────────────"
            f"\n  LSTM Karar - Gerçek Fail (t₁-t₀): {detection_delay_ms:+.1f} ms"
            f"\n  Migration Süresi         (t₂-t₁): {migration_ms:.1f} ms"
            f"{gain_line}"
            f"\n{'='*60}"
        )
        logger.warning(
            f"| Kurtarma Tamamlandı. "
            f"{action_count} switch migrate edildi."
        )
        logger.warning("====================================")
        import os
        os._exit(0)


    # ══════════════════════════════════════════════════════════════════════
    # Ana başlatıcı
    # ══════════════════════════════════════════════════════════════════════
    def run(self):
        logger.info(
            "SAEF-LSTM Proaktif Orkestratör Başlatıldı "
            f"({len(CTRL_DICTS)} controller, LSTM tabanlı hata algılama)"
        )

        # Thread-1: Proaktif SA planlama
        t_plan = threading.Thread(
            target=self.proactive_loop, daemon=True, name="SAPlanner"
        )
        t_plan.start()

        # Thread-2: Sadece c1 için LSTM izleme thread'i
        # (SAEF planlama thread'i tüm controller'ları kullanmaya devam eder)
        c1_dict   = CTRL_DICTS[0]
        c1_addr   = CONTROLLERS[0]
        t_lstm_c1 = threading.Thread(
            target=self.lstm_monitor_loop,
            args=(c1_addr, c1_dict),
            daemon=True,
            name="LSTM-c1"
        )
        t_lstm_c1.start()
        logger.info(f"[BAŞLATMA] {c1_dict['name']} LSTM izleme thread'i başlatıldı (tek controller).")
        monitor_threads = [t_lstm_c1]

        # Ana thread: monitor thread'lerinden en az biri bittiğinde uyar
        # (daemon olmayan thread yok, bu join ana süreci ayakta tutar)
        try:
            for t in monitor_threads:
                t.join()
        except KeyboardInterrupt:
            logger.info("Kullanıcı tarafından durduruldu.")


if __name__ == "__main__":
    SAEF_Orchestrator().run()