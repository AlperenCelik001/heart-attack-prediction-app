import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Model ve sütunları
model = joblib.load("kalp_modeli.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Kalp Krizi Riski", layout="centered")

if "tahmin_baslatildi" not in st.session_state:
    st.session_state["tahmin_baslatildi"] = False

if not st.session_state["tahmin_baslatildi"]:
    with st.form("kalp_formu"):
        st.title("🫠 Kalp Krizi Risk Analizi")
        st.subheader("Lütfen bilgilerinizi giriniz:")

        age = st.slider("Yaşınız", 20, 100, 50)
        sex = st.radio("Cinsiyet", ["Erkek", "Kadın"])

        cp_options = {
            0: "0 - Tipik anjina",
            1: "1 - Atipik anjina",
            2: "2 - Anjina dışı semptomlar",
            3: "3 - Ağrı yok"
        }
        cp = st.selectbox("Göğüs ağrısı tipi", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

        trestbps = st.number_input("İstirahat kan basıncı", 80, 200, 120)
        chol = st.number_input("Kolesterol (mg/dL)", 100, 600, 200)
        fbs = st.radio("Açlık kan şekeri > 120 mg/dL?", ["Evet", "Hayır"])

        restecg_options = {
            0: "0 - Normal",
            1: "1 - ST-T dalga anormalliği",
            2: "2 - Sol ventrikül hipertrofisi"
        }
        restecg = st.selectbox("Dinlenme EKG sonucu", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

        thalach = st.number_input("Maksimum kalp atım hızı", 60, 220, 150)
        exang = st.radio("Egzersize bağlı anjina?", ["Evet", "Hayır"])
        oldpeak = st.number_input("ST depresyonu", 0.0, 10.0, 1.0, step=0.1)

        slope_options = {
            0: "0 - Yatay",
            1: "1 - Yükselen",
            2: "2 - Alçalan"
        }
        slope = st.selectbox("ST segmentinin eğimi", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

        ca = st.selectbox("Floroskopiyle tespit edilen damar sayısı (0–3)", [0, 1, 2, 3])

        thal_options = {
            0: "0 - Normal",
            1: "1 - Sabit defekt",
            2: "2 - Geri dönüşlü defekt"
        }
        thal = st.selectbox("Thalium tarama sonucu", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

        tahmin_btn = st.form_submit_button("🔍 Tahmini Göster")

    if tahmin_btn:
        try:
            sex_val = 1 if sex == "Erkek" else 0
            fbs_val = 1 if fbs == "Evet" else 0
            exang_val = 1 if exang == "Evet" else 0

            user_input = {
                "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs_val, "restecg": restecg,
                "thalach": thalach, "exang": exang_val, "oldpeak": oldpeak,
                "slope": slope, "ca": ca, "thal": thal
            }

            df_user = pd.DataFrame([user_input])
            kategori_kolonlar = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
            for col in kategori_kolonlar:
                df_user[col] = df_user[col].astype(str)

            st.write("🔎 Girdi Verisi (Ham)", df_user)
            df_encoded = pd.get_dummies(df_user)

            for col in model_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0

            df_encoded = df_encoded[model_columns]
            st.write("📦 One-Hot Encoded Veri", df_encoded)

            prediction = model.predict(df_encoded)
            df_encoded["condition"] = prediction[0]

            try:
                df_existing = pd.read_csv("veri_kayitlari.csv")
                df_combined = pd.concat([df_existing, df_encoded], ignore_index=True)
            except FileNotFoundError:
                df_combined = df_encoded

            for col in model_columns:
                if col not in df_combined.columns:
                    df_combined[col] = 0

            df_combined = df_combined.fillna(0)
            df_combined.to_csv("veri_kayitlari.csv", index=False)

            st.session_state["tahmin_baslatildi"] = True
            st.session_state["prediction"] = int(prediction[0])

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

elif st.session_state["tahmin_baslatildi"]:
    st.empty()

    with st.container():
        with st.spinner("Nabız ölçülüyor..."):
            for _ in range(4):
                st.markdown("<h1 style='text-align:center;'>❤️</h1>", unsafe_allow_html=True)
                time.sleep(0.4)
                st.markdown("<h1 style='text-align:center;'>🖤</h1>", unsafe_allow_html=True)
                time.sleep(0.4)

    st.empty()
    with st.container():
        if st.session_state["prediction"] == 1:
            st.error("⚠️ Kalp krizi riski tespit edildi! Lütfen doktorunuza danışın.")
        else:
            st.success("🟢 Kalp krizi riski düşük görünüyor. Yine de düzenli kontroller önemlidir.")

    st.markdown("---")

    try:
        df_vis = pd.read_csv("veri_kayitlari.csv")

        st.markdown("## 📊 Analizler ve Görselleştirme")

        # 🎯 Confusion Matrix (Doğruluk)
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix

        X = df_vis.drop("condition", axis=1)
        y = df_vis["condition"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=["Düşük", "Yüksek"],
                    yticklabels=["Düşük", "Yüksek"])
        ax_cm.set_title("🎯 Model Doğruluk Matrisi")
        ax_cm.set_xlabel("Tahmin")
        ax_cm.set_ylabel("Gerçek")
        st.pyplot(fig_cm)

        # Kalp krizi tahmin dağılımı
        tahmin_sayilari = df_vis["condition"].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        ax1.bar(["Düşük Risk (0)", "Yüksek Risk (1)"], tahmin_sayilari, color=["green", "red"])
        ax1.set_title("Kalp Krizi Tahmin Dağılımı")
        st.pyplot(fig1)

        # Yaş histogramı
        fig2, ax2 = plt.subplots()
        sns.histplot(df_vis["age"], kde=True, ax=ax2, color="skyblue")
        ax2.set_title("Yaş Dağılımı")
        st.pyplot(fig2)

        # Cinsiyet oranı
        cinsiyet_kolonlari = [col for col in df_vis.columns if col.startswith("sex_")]
        if cinsiyet_kolonlari:
            cinsiyet_verisi = df_vis[cinsiyet_kolonlari].sum()
            fig3, ax3 = plt.subplots()
            ax3.pie(cinsiyet_verisi, labels=cinsiyet_kolonlari, autopct="%1.1f%%", startangle=90)
            ax3.set_title("Cinsiyet Dağılımı")
            st.pyplot(fig3)

        # Risk oranı bilgisi
        toplam = len(df_vis)
        pozitif = df_vis["condition"].sum()
        oran = pozitif / toplam * 100
        st.info(f"Toplam kayıt: {toplam} | Yüksek risk oranı: %{oran:.2f}")

    except Exception as e:
        st.warning(f"📉 Görselleştirme hatası: {e}")

    st.button("↩️ Yeni Tahmin Yap", on_click=lambda: st.session_state.update({
        "tahmin_baslatildi": False,
        "df_encoded": None,
        "prediction": None
    }))
