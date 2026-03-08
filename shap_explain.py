"""
shap_explain.py
Stres veya Fail tespiti yapıldığında, hangi feature'ların
bu kararı ne kadar etkilediğini açıklar (XAI).
"""
import numpy as np
import torch
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Konfigürasyon ─────────────────────────────────────────────
FEATURES = ['cpu_usage', 'memory_usage', 'heap_mb',
            'heap_growth_bps', 'proc_cpu_pct', 'rest_rtt_ms']
LABEL_NAMES = ['Normal', 'Stress', 'Fail']

# ── Lazy import ───────────────────────────────────────────────
_explainer = None
_scaler    = None
_model     = None

def _get_model():
    """Model'i gerektiğinde yükle (lazy load)."""
    global _model
    if _model is None:
        from lstm_train import LSTMModel
        _model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, output_size=3)
        _model.load_state_dict(
            torch.load("lstm_model.pth", map_location="cpu"))
        _model.eval()
    return _model

def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load("scaler.pkl")
    return _scaler

def _model_predict_flat(X_flat):
    """
    KernelExplainer için düz (batch, seq*features) girişi alır,
    (batch, n_classes) olasılık matrisi döner.
    """
    model = _get_model()
    seq_len = 3   # lstm_model.pth ile aynı olmalı
    n_feat  = len(FEATURES)
    X_3d = X_flat.reshape(-1, seq_len, n_feat)
    X_t  = torch.tensor(X_3d, dtype=torch.float32)
    with torch.no_grad():
        out  = model(X_t)
        prob = torch.softmax(out, dim=1).numpy()
    return prob

def build_explainer(background_X_3d):
    """
    background_X_3d: (N, seq_len, n_features) — training'den alınmış örnekler.
    Explainer'ı bir kez oluştur.
    """
    global _explainer
    import shap
    bg_flat = background_X_3d.reshape(len(background_X_3d), -1)
    # KernelExplainer tek seferde kurulur, sonra hızlı çalışır.
    _explainer = shap.KernelExplainer(_model_predict_flat, bg_flat)
    return _explainer

def explain_prediction(X_sequence_3d, predicted_label):
    """
    X_sequence_3d : (1, seq_len, n_features)  — Son 10 adım, normalize edilmiş.
    predicted_label: 0, 1 veya 2
    Konsola hangi feature'ın kararı ne kadar etkilediğini yazar.
    """
    global _explainer
    if _explainer is None:
        print("[XAI] Explainer henüz kurulmadı, bu tahmin atlanıyor.")
        return

    import shap
    X_flat = X_sequence_3d.reshape(1, -1)
    # nsamples=50 ile hızlı yaklaşık hesaplama
    shap_values = _explainer.shap_values(X_flat, nsamples=50, l1_reg="num_features(6)")

    # SHAP versiyonuna göre çıktı formatı değişebilir:
    # - Eski: list[n_classes] of (n_samples, n_features)  → shap_values[class][0]
    # - Yeni: (n_samples, n_features) tek array           → shap_values[0]
    if isinstance(shap_values, list):
        if len(shap_values) > predicted_label:
            class_shap = shap_values[predicted_label][0]
        else:
            class_shap = shap_values[0][0]  # fallback
    else:
        # Tek array → tüm sınıflar için birleşik değer (genellikle en büyük sınıfa ait)
        class_shap = np.array(shap_values).reshape(-1)

    # Son 3 zaman adımına bak (en güncel ve en bağlamsal veriler)
    seq_len = 3   # lstm_model.pth ile aynı olmalı
    n_feat  = len(FEATURES)
    total_per_class = seq_len * n_feat  # 60

    # Eğer SHAP 3 sınıfın hepsini tek array'e koymuşsa (180 = 3*10*6),
    # sadece tahmin edilen sınıfa ait dilimi al.
    if len(class_shap) == 3 * total_per_class:
        class_shap = class_shap[predicted_label * total_per_class :
                                 (predicted_label + 1) * total_per_class]

    # Reshape'e güvenli gir
    class_shap = class_shap[:total_per_class]  # her ihtimale karşı
    last_steps = class_shap.reshape(seq_len, n_feat)[-3:]  # (3, 6)
    avg_shap   = last_steps.mean(axis=0)                    # (6,)

    print(f"\n  ── XAI: [{LABEL_NAMES[predicted_label]}] kararının nedenleri ────────")
    for feat, val in sorted(zip(FEATURES, avg_shap), key=lambda x: abs(x[1]), reverse=True):
        direction = "▲" if val > 0 else "▼"
        bar_len = int(abs(val) * 200)
        bar = "█" * min(bar_len, 30)
        print(f"  {direction} {feat:<22s} {val:+.4f}  {bar}")
    print(f"  ───────────────────────────────────────────────")
