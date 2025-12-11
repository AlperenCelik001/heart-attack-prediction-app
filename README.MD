# Heart Attack Prediction App

This project is a **machine learning–based decision support tool** that estimates the **risk of heart attack** from basic clinical features and presents the result through an **interactive Streamlit web interface**.

---
Project Overview:

- Uses the well-known **heart disease dataset** (Cleveland-based CSV).
- Trains a **Random Forest Classifier** and reaches around  
  **~0.82 accuracy** on the test set.
- Saves the trained model and feature names with **joblib**.
- Provides a **Streamlit UI** where the user:
  - Enters age, chest-pain type, cholesterol, ST depression, etc.
  - Gets an instant **risk prediction** (low / high).
  - Sees **basic visualizations** like:
    - Confusion matrix heatmap
    - Risk distribution
    - Age and gender statistics of saved predictions

---

 Machine Learning:

Machine Learning

- **Model:** `RandomForestClassifier` (sklearn)
- **Training script:** `Hearth.py`
- **Target variable:** `condition` (0 = low risk, 1 = high risk)
- **Saved artifacts:**
  - `kalp_modeli.pkl` – trained Random Forest model
  - `model_columns.pkl` – list of one-hot encoded feature names
- **Dynamic dataset:**
  - User inputs and model predictions are appended to  
    `veri_kayitlari.csv`, so the dataset grows over time.

---

Streamlit Application:

- **Main app:** `kalp_analiz_app.py`
- Collects user information (age, gender, chest pain type, etc.).
- Encodes the inputs using the same one-hot scheme as in training.
- Loads `kalp_modeli.pkl` and predicts the risk.
- Shows:
  - Animated heartbeat effect while prediction is running
  - Clear message:
    -  “Low risk”  
    -  “High risk – please consult your doctor”
  - Confusion matrix, class distribution and simple EDA plots.


### Run the app locally

```bash
# 1. (Optional) create venv and install dependencies
pip install -r requirements.txt  # yoksa: pip install streamlit scikit-learn pandas matplotlib seaborn joblib

# 2. Streamlit uygulamasını başlat
streamlit run kalp_analiz_app.py

Then open the URL shown in the terminal (usually http://localhost:8501).
