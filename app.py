import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Chargement des artefacts
ARTIFACT_DIR = "churn_ann_artifacts_v2"

preprocess = joblib.load(os.path.join(ARTIFACT_DIR, "preprocess.joblib"))

# Détection du type de modèle
model_path_h5 = os.path.join(ARTIFACT_DIR, "ann_model.h5")
model_path_joblib = os.path.join(ARTIFACT_DIR, "mlp_model.joblib")

use_tf = False
if os.path.exists(model_path_h5):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path_h5)
    use_tf = True
else:
    model = joblib.load(model_path_joblib)

# Interface
st.set_page_config(page_title="Churn Client – Télécom", layout="centered")
st.title("📊 Prédiction du churn client")
st.markdown(
    "Cette application estime le **risque de résiliation** d’un client à partir de ses caractéristiques."
)

st.header("🧾 Informations client")

# Formulaire
geography = st.selectbox("Pays", ["France", "Spain", "Germany"])
gender = st.selectbox("Genre", ["Male", "Female"])
age = st.slider("Âge", 18, 100, 40)
tenure = st.slider("Ancienneté (années)", 0, 10, 5)
balance = st.number_input("Solde du compte", min_value=0.0, value=50000.0)
num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4])
has_cr_card = st.selectbox("Carte de crédit", [0, 1])
is_active = st.selectbox("Client actif", [0, 1])
credit_score = st.slider("Score de crédit", 300, 900, 650)
estimated_salary = st.number_input("Salaire estimé", min_value=0.0, value=60000.0)

# Bouton de prédiction
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

    X = preprocess.transform(input_df)

    try:
        X = X.toarray()
    except Exception:
        pass

    if use_tf:
        proba = model.predict(X)[0][0]
    else:
        proba = model.predict_proba(X)[0][1]

    st.subheader("📈 Résultat")
    st.write(f"**Probabilité de churn : {proba:.2%}**")

    if proba >= 0.5:
        st.error("⚠️ Client à risque de résiliation")
    else:
        st.success("✅ Client à faible risque")
