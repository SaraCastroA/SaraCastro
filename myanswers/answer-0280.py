import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def analizar_confianza_lector(df):

    X = df[[
        "num_paginas_leidas",
        "frecuencia_lectura",
        "nivel_comprension"
    ]]

    y = df["habito"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression()
    modelo.fit(X_scaled, y)

    probs = modelo.predict_proba(X_scaled)

    max_probs = probs.max(axis=1)

    indices_baja_confianza = np.where(max_probs < 0.6)[0]

    return indices_baja_confianza
