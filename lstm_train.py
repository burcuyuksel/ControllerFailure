import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Veri yükleme ──────────────────────────────────────────────
data = pd.read_csv("dataset_7mart.csv")

# Tüm feature sütunları (timestamp ve label hariç)
features = ['cpu_usage', 'memory_usage', 'heap_mb', 'heap_growth_bps',
            'proc_cpu_pct', 'rest_rtt_ms']
labels_col = 'label'

print(f"Dataset boyutu: {len(data)} satır, {len(features)} feature")
print(f"Label dağılımı:\n{data[labels_col].value_counts().sort_index()}\n")

# ── Normalizasyon ─────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features])

data_scaled = pd.DataFrame(scaled_features, columns=features)
data_scaled[labels_col] = data[labels_col].values

# ── Sekans oluşturma (LSTM için) ─────────────────────────────
# ÖNEMLI: Sequence'lar test sınırlarını AŞMAMALI.
# Ardışık satırlar arasında >30 saniye fark varsa yeni test başlıyor.
sequence_length = 5   # 3→5: FAIL geçiş örüntüsünü daha uzun pencereyle öğren
X, y = [], []

data['timestamp'] = pd.to_datetime(data['timestamp'])
time_diffs = data['timestamp'].diff().dt.total_seconds().fillna(0)
# 30 saniyeden büyük fark = test sınırı → o indekste yeni grup başlar
group_starts = [0] + list(data.index[time_diffs > 30])
group_starts.append(len(data))

for g in range(len(group_starts) - 1):
    start, end = group_starts[g], group_starts[g + 1]
    group = data_scaled.iloc[start:end]
    for i in range(len(group) - sequence_length):
        X.append(group.iloc[i:i + sequence_length][features].values)
        y.append(group.iloc[i + sequence_length][labels_col])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"Sekans sayısı: {len(X)}, Sekans uzunluğu: {sequence_length}")
print(f"Test grubu sayısı: {len(group_starts) - 1}")

# ── Train / Test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── Class weights (dengesiz veri için) ────────────────────────
class_counts = np.bincount(y_train)
total = len(y_train)

# Standart class weight hesaplama
base_weights = [total / (len(class_counts) * c) if c > 0 else 0.0 for c in class_counts]

class_weights = torch.tensor(base_weights, dtype=torch.float32)
print(f"Custom Class weights: {class_weights.tolist()}")

# ── DataLoader ────────────────────────────────────────────────
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ── LSTM Model ────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out

# ── Hiperparametreler ─────────────────────────────────────────
input_size  = len(features)      # 6
hidden_size = 64                 # 128→ 64 (297 seq için daha uygun)
num_layers  = 2
output_size = len(data_scaled[labels_col].unique())  # 3
learning_rate = 0.001
num_epochs    = 150
patience      = 35               # 20 → 35

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# ── Model, loss, optimizer, scheduler ───────────────────────
model     = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout=0.4).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

# ── Eğitim döngüsü ───────────────────────────────────────────
best_acc      = 0.0
no_improve    = 0
print(f"\nEğitim başlıyor ({num_epochs} epoch, early_stop patience={patience})...\n")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Her epoch test accuracy (scheduler ve early stopping için gerekli)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    acc = 100 * correct / total
    scheduler.step(acc)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}]  Loss: {avg_loss:.4f}  Test Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        no_improve = 0
        torch.save(model.state_dict(), "lstm_model.pth")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"  → Early stopping epoch {epoch+1} (patience={patience})")
        break

# ── Son test ve detaylı rapor ─────────────────────────────────
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

label_names = ['Normal/Open', 'Stress', 'Fail']
print(f"\n{'='*50}")
print(f"En iyi Test Accuracy: {best_acc:.2f}%")
print(f"{'='*50}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print(f"\nModel kaydedildi: lstm_model.pth")

# Scaler'ı da kaydet (real-time prediction için)
import joblib
joblib.dump(scaler, "scaler.pkl")
print("Scaler kaydedildi: scaler.pkl")