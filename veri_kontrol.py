import pandas as pd

# CSV'den oku
df = pd.read_csv("veri_kayitlari.csv")

# Kayıt sayısı ve son 5 satırı yazdır
print("Toplam kayıt sayısı:", len(df))
print(df.tail(5))  # Son 5 veri

# Son eklenen (en alt) satırı açıkça yazdır
print("\n📌 Son eklenen tam kayıt:")
print(df.iloc[-1].to_string())


#import pandas as pd

#df = pd.read_csv("veri_kayitlari.csv")
#print("Toplam kayıt sayısı:", len(df))
#print(df.tail(5))  # son 5 veri
