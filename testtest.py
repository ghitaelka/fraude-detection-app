
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import io
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# 🔐 AUTHENTIFICATION SIMPLE
# =====================

st.sidebar.title("🔐 Connexion")
users = {
    "admin": "1234",
    "hps_user": "securepass"
}

username = st.sidebar.text_input("Nom d'utilisateur")
password = st.sidebar.text_input("Mot de passe", type="password")

if username in users and users[username] == password:
    st.success(f"Bienvenue {username} 👋")

    # =====================
    #  PARTIE ENTRAÎNEMENT
    # =====================

    df = pd.read_csv("creditcard.csv")
    df = df.dropna(axis=1, how='all').fillna(0)

    for col in df.columns:
        if "DATE" in col.upper():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].astype("int64") // 10**9
            except:
                pass

    df = pd.get_dummies(df, drop_first=True)
    df = df.select_dtypes(include=["number", "bool"])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    joblib.dump((lr, X.columns.tolist(), scaler), "creditcard.pkl")

    # =====================
    # PARTIE STREAMLIT
    # =====================

    st.title("💳 Détection de Fraude ATM")

    uploaded_file = st.file_uploader("📁 Uploade un fichier CSV avec les transactions :", type="csv")

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.subheader("Aperçu des données")
        st.dataframe(df_input.head())

        model, expected_columns, scaler = joblib.load("creditcard.pkl")

        df_input = df_input.dropna(axis=1, how='all').fillna(0)

        for col in df_input.columns:
            if "DATE" in col.upper():
                try:
                    df_input[col] = pd.to_datetime(df_input[col], errors="coerce")
                    df_input[col] = df_input[col].astype("int64") // 10**9
                except:
                    pass

        df_input = pd.get_dummies(df_input, drop_first=True)
        df_input = df_input.select_dtypes(include=["number", "bool"])

        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[expected_columns]
        df_input_scaled = scaler.transform(df_input)

        if st.button("🔍 Prédire les fraudes"):
            prediction = model.predict(df_input_scaled)
            df_input["Fraude_Prédit"] = prediction

            nb_fraudes = (df_input["Fraude_Prédit"] == 1).sum()
            st.markdown(f"✅ **{nb_fraudes} fraudes détectées sur {len(df_input)} transactions.**")

            st.subheader("📊 Résultats complets")
            st.dataframe(df_input)

            def highlight_fraudes(row):
                return ['background-color: #ffcccc' if row.Fraude_Prédit == 1 else '' for _ in row]

            st.subheader("⚠️ Transactions suspectes")
            st.dataframe(df_input[df_input["Fraude_Prédit"] == 1].style.apply(highlight_fraudes, axis=1))



else:
    st.warning("Veuillez entrer vos identifiants pour accéder à l'application.")
