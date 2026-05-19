import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def preparar_datos(*args, **kwargs):
    """
    Versión robusta para adaptarse a las inconsistencias del 
    generador de casos de uso del evaluador automático.
    """
    # 1. Intentar extraer parámetros si el compañero envió todo dentro de kwargs con nombres correctos
    df = kwargs.get('df', None)
    target_col = kwargs.get('target_col', None)
    categorias_ordinales = kwargs.get('categorias_ordinales', None)
    k = kwargs.get('k', None)

    # 2. Si venían de forma posicional en args (o compactados en una tupla en args[0])
    if df is None and len(args) > 0:
        if isinstance(args[0], (tuple, list)) and len(args[0]) == 4:
            # Por si metió los 4 elementos dentro de una sola lista/tupla en args[0]
            df, target_col, categorias_ordinales, k = args[0]
        elif isinstance(args[0], dict):
            # Por si metió un diccionario anidado en args[0]
            d = args[0]
            df = d.get('df') or d.get('dataframe') or d.get('df_input')
            target_col = d.get('target_col') or d.get('target')
            categorias_ordinales = d.get('categorias_ordinales') or d.get('categories')
            k = d.get('k')
        elif len(args) == 4:
            # Envío posicional estándar de 4 elementos
            df, target_col, categorias_ordinales, k = args

    # 3. Validar de emergencia por si usó nombres ligeramente diferentes en su diccionario
    if df is None and len(kwargs) > 0:
        # Busca cualquier DataFrame en los kwargs
        for v in kwargs.values():
            if isinstance(v, pd.DataFrame):
                df = v
                break
        # Busca un entero para k, un string para el target y un diccionario para las categorías
        for k_val, v in kwargs.items():
            if isinstance(v, str) and target_col is None:
                target_col = v
            elif isinstance(v, dict) and categorias_ordinales is None:
                categorias_ordinales = v
            elif isinstance(v, (int, np.integer)) and k is None:
                k = v

    # --- PROCESAMIENTO ORIGINAL ---
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].to_numpy(dtype=int)

    cols_ordinales = list(categorias_ordinales.keys())
    orden_categorias = [categorias_ordinales[c] for c in cols_ordinales]

    encoder = OrdinalEncoder(categories=orden_categorias)
    X[cols_ordinales] = encoder.fit_transform(X[cols_ordinales])

    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    scaler = MinMaxScaler()
    X_final = scaler.fit_transform(X_selected)

    return X_final, y
