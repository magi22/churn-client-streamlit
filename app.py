import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# =========================
# Chargement des fichiers
# =========================

# Répertoire racine du projet (là où se trouve app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESS_PATH = os.path.join(BASE_DIR, "preprocess.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "mlp_model.joblib")

# Vérifications de sécurité (messages clairs côté Streamlit)
if not os.path.exists(PREPROCESS_PATH):
    st.error("❌ Fichier preprocess.joblib introuvable dans le dépôt.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("❌ Fichier mlp_model.joblib introuvable dans le dépôt.")
    st.stop()

# Chargement
preprocess = joblib.load(PREPROCESS_PATH)
model = joblib.load(MODEL_PATH)

# =========================
# Interface Streamlit
# =========================

st.set_page_config(page_title="Churn Client – Télécom", layout="centered")

st.title("📊 Prédiction du churn client")
st.markdown(
    "Cette application estime le **risque de résiliation** d’un client à partir de ses caractéristiques."
)

st.header("🧾 Informations client")

# =========================
# Formulaire utilisateur
# =========================

geography = st.selectbox("Pays", ["France", "Spain", "Germany"])
gender = st.selectbox("Genre", ["Male", "Female"])
age = st.slider("Âge", 18, 100, 40)
tenure = st.slider("Ancienneté (années)", 0, 10, 5)
balance = st.number_input("Solde du compte", min_value=0.0, value=50000.0)
num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4])
has_cr_card = st.selectbox(
    "Carte de crédit",
    [0, 1],
    help="0 = le client ne possède pas de carte de crédit | 1 = le client possède une carte de crédit"
)

is_active = st.selectbox(
    "Client actif",
    [0, 1],
    help="0 = client peu ou pas actif | 1 = client actif (utilisation régulière des services)"
)
credit_score = st.slider("Score de crédit", 300, 900, 650)
estimated_salary = st.number_input("Salaire estimé", min_value=0.0, value=60000.0)

# =========================
# Prédiction
# =========================

if st.button("🔍 Estimer le risque"):
    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary
    }])

    # Prétraitement
    X = preprocess.transform(input_df)

    try:
        X = X.toarray()
    except Exception:
        pass

    # Prédiction (probabilité churn)
    proba = model.predict_proba(X)[0][1]

    st.subheader("📈 Résultat")
    st.write(f"**Probabilité de churn : {proba:.2%}**")

    if proba >= 0.5:
        st.error("⚠️ Client à risque de résiliation")
    else:
        st.success("✅ Client à faible risque")

