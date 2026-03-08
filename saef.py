import requests
import time
import math
import random
import logging
import threading  # Thread yapısı için eklendi

# --- Yapılandırma ---
CONTROLLERS = ["192.168.56.107:6653", "192.168.56.107:7753", "192.168.56.107:8853"]
REST_PORTS = ["8080", "8081", "8082"]
SWITCHES = [f"00:00:00:00:00:00:00:0{i}" for i in range(1, 4)]

# DPID -> Kısa İsim eşleşmesi
SW_NAMES = {f"00:00:00:00:00:00:00:0{i}": f"s{i}" for i in range(1, 4)}
# Controller -> Kısa İsim eşleşmesi
CP_NAMES = {CONTROLLERS[0]: "c1", CONTROLLERS[1]: "c2", CONTROLLERS[2]: "c3"}

TIMEOUT = 2.1
#Makaledeki hiyerarşik yapıda, bir denetleyicinin öldüğüne karar verilmeden önce genellikle en az iki ardışık kalp atışının kaçırılması beklenir (2x1000 ms=2000 ms).
# Üzerine eklenen 0.1 saniyelik pay, ağdaki anlık dalgalanmaları (jitter) tolere etmek içindir.
HEARTBEAT_INTERVAL = 1.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAEF_DEBUG")

class TwoWindowFD:
    def __init__(self, sw_size=10, lw_size=100, safety_margin=1.5):
        self.short_window = [] # Anlık değişimler
        self.long_window = []  # Genel ağ karakteri
        self.sw_size = sw_size
        self.lw_size = lw_size
        self.safety_margin = safety_margin

    def add_interval(self, interval):
        self.short_window.append(interval)
        self.long_window.append(interval)
        if len(self.short_window) > self.sw_size: self.short_window.pop(0)
        if len(self.long_window) > self.lw_size: self.long_window.pop(0)

    def get_dynamic_timeout(self):
        if not self.short_window or not self.long_window:
            return 2.0
        avg_lw = sum(self.long_window) / len(self.long_window)
        variance_sw = sum((x - (sum(self.short_window)/len(self.short_window)))**2
                          for x in self.short_window) / len(self.short_window)
        std_dev_sw = math.sqrt(variance_sw)
        return avg_lw + (self.safety_margin * std_dev_sw)

class SAEF_Orchestrator:
    def __init__(self):
        self.active_controllers = list(CONTROLLERS)
        self.plan_bank = {}
        self.controller_status = {cp: time.time() for cp in CONTROLLERS}
        self.load_data = {cp: 0.1 for cp in CONTROLLERS}
        self.alpha = 0.5
        self.beta = 0.5
        self.lbm_history = []
        self.fd_monitors = {cp: TwoWindowFD() for cp in CONTROLLERS}
        self.last_heartbeat_time = {cp: time.time() for cp in CONTROLLERS}
        self.lock = threading.Lock() # Plan bankasına erişim güvenliği için

    def get_sw_name(self, dpid): return SW_NAMES.get(dpid, dpid)
    def get_cp_name(self, cp): return CP_NAMES.get(cp, cp)

    def log_system_status(self):
        curr_lbm, avg_lbm = self.calculate_lbm()
        d_avg = self.calculate_avg_propagation_delay()
        t_proc = sum([self.calculate_avg_processing_time(cp) for cp in self.active_controllers]) / len(self.active_controllers)
        logger.info(f"=== [SİSTEM DURUMU] ===")
        logger.info(f"| Aktif Denetleyiciler: {len(self.active_controllers)}")
        for cp in self.active_controllers:
            logger.info(f"|-- {self.get_cp_name(cp)} Yük: {self.load_data[cp]:.4f}")
        logger.info(f"| Avg Propagation Delay (D_avg): {d_avg:.2f} ms")
        logger.info(f"| Avg Processing Time (T_proc): {t_proc:.2f} ms")
        logger.info(f"| Mevcut LBM: {curr_lbm:.3f} | Ortalama LBM: {avg_lbm:.3f}")
        logger.info(f"| Mevcut Ağırlıklar -> Alpha (Gecikme): {self.alpha:.2f}, Beta (Yük): {self.beta:.2f}")
        logger.info(f"========================")

    def calculate_lbm(self):
        loads = [self.load_data[cp] for cp in self.active_controllers]
        if not loads: return 0, 0
        avg_load = sum(loads) / len(loads)
        curr_lbm = max(loads) / avg_load if avg_load > 0 else 1.0
        self.lbm_history.append(curr_lbm)
        return curr_lbm, sum(self.lbm_history) / len(self.lbm_history)

    def get_latency(self, sw_id, target_cp):
        try:
            target_idx = CONTROLLERS.index(target_cp)
            port = REST_PORTS[target_idx]
            url = f"http://192.168.56.107:{port}/wm/core/switch/{sw_id}/desc/json"
            start_time = time.time()
            response = requests.get(url, timeout=0.5)
            if response.status_code == 200:
                return max((time.time() - start_time) * 500.0, 0.1)
            return 100.0
        except: return 200.0

    def calculate_avg_processing_time(self, cp):
        return 1.0 + (self.load_data[cp] * 10.0)

    def calculate_avg_propagation_delay(self):
        total_lat, count = 0, 0
        for sw in SWITCHES:
            for cp in self.active_controllers:
                total_lat += self.get_latency(sw, cp)
                count += 1
        return total_lat / count if count > 0 else 5.0

    def adjust_weights_proactively(self):
        d_avg = self.calculate_avg_propagation_delay()
        t_proc = sum([self.calculate_avg_processing_time(cp) for cp in self.active_controllers]) / len(self.active_controllers)
        ratio = d_avg / t_proc
        if ratio >= 2: self.alpha = 1.0
        elif ratio > 0: self.alpha = ratio * 0.5
        self.beta = 1.0 - self.alpha
        logger.info(f"[DYNAMIC WEIGHTS] D_avg: {d_avg:.2f}, T_proc: {t_proc:.2f}, Ratio: {ratio:.2f} -> Alpha: {self.alpha:.2f}")

    def cost_function(self, plan, current_active_list):
        total_lat = sum([self.get_latency(sw, cp) for sw, cp in plan.items()])
        norm_delay = min((total_lat / len(plan)) / 20.0, 1.0)
        sim_loads = {cp: 0 for cp in current_active_list}
        for cp in plan.values(): sim_loads[cp] += 1
        loads = list(sim_loads.values())
        mean = sum(loads) / len(loads)
        lsd = math.sqrt(sum([((x - mean) ** 2) for x in loads]) / len(loads))
        norm_lsd = lsd / 3.0
        return (self.alpha * norm_delay) + (self.beta * norm_lsd)

    def run_sa_for_scenario(self, scenario_name, scenario_active_list):
        logger.info(f"[{scenario_name} - SA BAŞLADI] Hesaplanıyor...")
        active_list = list(scenario_active_list)
        T_e = 100.0
        current_plan = {sw: random.choice(active_list) for sw in SWITCHES}
        current_cost = self.cost_function(current_plan, active_list)
        initial_cost = current_cost
        accepted_worse = 0
        for t in range(50):
            neighbor_plan = current_plan.copy()
            sw1, sw2 = random.sample(SWITCHES, 2)
            neighbor_plan[sw1], neighbor_plan[sw2] = neighbor_plan[sw2], neighbor_plan[sw1]
            neighbor_cost = self.cost_function(neighbor_plan, active_list)
            delta_c = neighbor_cost - current_cost
            if neighbor_cost < current_cost or random.random() < math.exp(-delta_c / T_e):
                current_plan, current_cost = neighbor_plan, neighbor_cost
                if delta_c > 0: accepted_worse += 1
            T_e *= 0.95
        pretty_plan = {self.get_sw_name(sw): self.get_cp_name(cp) for sw, cp in current_plan.items()}
        logger.info(f"[{scenario_name} - SA BİTTİ] Maliyet: {initial_cost:.4f} -> {current_cost:.4f}| Atama: {pretty_plan}")
        return current_plan

    # --- Thread 1: Proaktif Hesaplama Döngüsü ---
    def proactive_loop(self):
        while True:
            self.log_system_status()
            self.adjust_weights_proactively()
            logger.info(f"[PLAN BANKASI GÜNCELLEME]")
            new_plans = {}
            new_plans["NORMAL"] = self.run_sa_for_scenario("NORMAL", self.active_controllers)
            if len(self.active_controllers) > 1:
                for cp in self.active_controllers:
                    hypo = [c for c in self.active_controllers if c != cp]
                    new_plans[f"FAIL_{cp}"] = self.run_sa_for_scenario(f"{self.get_cp_name(cp)} ÇÖKERSE", hypo)
            with self.lock:
                self.plan_bank = new_plans
            time.sleep(3) # Planları her 3 saniyede bir güncellemek yeterli

    # --- Thread 2: Hata Algılama Döngüsü ---
    def check_failures_loop(self):
        while True:
            for cp in list(self.active_controllers):
                idx = CONTROLLERS.index(cp)
                load = self.get_load(f"192.168.56.107:{REST_PORTS[idx]}")
                if load is not None:
                    arrival_interval = time.time() - self.controller_status[cp]
                    self.fd_monitors[cp].add_interval(arrival_interval)
                    self.controller_status[cp] = time.time()
                    self.load_data[cp] = load
                else:
                    dynamic_threshold = self.fd_monitors[cp].get_dynamic_timeout()
                    dynamic_threshold = max(dynamic_threshold, HEARTBEAT_INTERVAL * 1.1)
                    time_diff = time.time() - self.controller_status[cp]
                    if time_diff > dynamic_threshold:
                        logger.error(f"!!! 2W-FD KARARI: {self.get_cp_name(cp)} ÖLDÜ !!!")
                        self.active_controllers.remove(cp)
                        detection_start_ref = self.controller_status[cp]
                        self.trigger_recovery(cp, detection_start_ref)
            time.sleep(HEARTBEAT_INTERVAL)

    def get_load(self, url):
        try:
            r = requests.get(f"http://{url}/wm/statistics/bandwidth/all/all/json", timeout=0.4)
            return max(sum(int(d['bits-per-second-rx']) for d in r.json()) / 5000.0, 0.05)
        except: return None

    def trigger_recovery(self, failed_cp, start_time):
        logger.warning(f"=== [FAILOVER AKSİYONU BAŞLADI] ===")
        with self.lock:
            recovery_plan = self.plan_bank.get(f"FAIL_{failed_cp}")
        if not recovery_plan: return
        action_count = 0
        for sw, target_cp in recovery_plan.items():
            try:
                target_idx = CONTROLLERS.index(target_cp)
                url = f"http://192.168.56.107:{REST_PORTS[target_idx]}/wm/core/switch/{sw}/role/json"
                requests.post(url, json={"role": "MASTER"}, timeout=0.5)
                logger.info(f"|-- [SAEF KURTARMA] {self.get_sw_name(sw)} -> {self.get_cp_name(target_cp)}")
                action_count += 1
            except: pass
        logger.warning(f"| Kurtarma Tamamlandı. Süre: {(time.time() - start_time)*1000:.2f} ms")

    def run(self):
        logger.info("SAEF Proaktif Orkestratör Başlatıldı (Multithread).")
        t1 = threading.Thread(target=self.proactive_loop, daemon=True)
        t2 = threading.Thread(target=self.check_failures_loop)
        t1.start()
        t2.start()
        t2.join()

if __name__ == "__main__":
    SAEF_Orchestrator().run()
