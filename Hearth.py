import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

print("Klasördeki dosyalar:", os.listdir())

# 1. Yeni veri dosyasını oku (Zaten encode'lu!)
df = pd.read_csv("veri_kayitlari.csv")
print("İlk 5 satır:")
print(df.head())

# 2. Özellik ve hedefi ayır
X = df.drop("condition", axis=1)
y = df["condition"]

# 3. Train/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Tahmin ve sonuç
y_pred = model.predict(X_test)
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Kaydet
joblib.dump(model, "kalp_modeli.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
print("✅ Model ve sütunlar kaydedildi.")

# 7. Dosyayı yedekle (isteğe bağlı)
if not os.path.exists("veri_kayitlari.csv"):
    df.to_csv("veri_kayitlari.csv", index=False)
    print("İlk veri kümesi veri_kayitlari.csv olarak kaydedildi.")

print("Son 5 kayıt:")
print(df.tail())
