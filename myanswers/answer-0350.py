# answer-0350.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

def preparar_datos(df, target_col, categorias_ordinales, k):
    """
    Prepara datos con variables ordinales:
    Codificación -> Selección de k mejores -> Escalado MinMax
    """

    # 1. Separar X e y
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].to_numpy(dtype=int)

    # 2. Codificación ordinal preservando el orden lógico definido en el dict
    cols_ordinales = list(categorias_ordinales.keys())
    orden_categorias = [categorias_ordinales[c] for c in cols_ordinales]

    encoder = OrdinalEncoder(categories=orden_categorias)
    X[cols_ordinales] = encoder.fit_transform(X[cols_ordinales])

    # 3. Selección de las k mejores features (basado en ANOVA f_classif)
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # 4. Normalización al rango [0, 1]
    scaler = MinMaxScaler()
    X_final = scaler.fit_transform(X_selected)

    return X_final, y
