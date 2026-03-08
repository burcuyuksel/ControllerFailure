import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("dataset_c1.csv")

# Etiket değiştirme kuralları
# Sadece Label 1 (Stress) olanları inceleyeceğiz. 0 ve 2'ye DOKUNMUYORUZ.
# Bir kaydın "Stress (1)" kalabilmesi için CPU veya Heap Growth vb. belirgin bir yük altında olmalı.

def reevaluate_stress(row):
    if row['label'] != 1:
        return row['label'] # 0 ve 2 aynen kalır

    # Stress (1) olanları kontrol et:
    # Eğer controller CPU'su %30'dan küçükse
    # VE sistem CPU'su %40'tan küçükse
    # VE RTT 15ms'den küçükse 
    # Bu aslında henüz Stress'e girmemiştir, Normal'dir (0).
    
    if row['proc_cpu_pct'] < 30.0 and row['cpu_usage'] < 40.0 and row['rest_rtt_ms'] < 15.0:
        return 0
        
    # Aksi halde bu gerçekten bir stres anıdır
    return 1


print("Re-evaluating 'Stress' (1) labels...")
print(f"Eski dağılım:\n{df['label'].value_counts().sort_index()}")

df['label'] = df.apply(reevaluate_stress, axis=1)

print(f"\nYeni dağılım:\n{df['label'].value_counts().sort_index()}")

# Üzerine yaz
df.to_csv("dataset_c1.csv", index=False)
print("\ndataset_c1.csv güncellendi.")
