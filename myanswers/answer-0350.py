from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def preparar_datos(df, target_col, categorias_ordinales, k):

    X = df.drop(columns=[target_col]).copy()

    y = df[target_col].to_numpy(dtype=int)

    cols_ordinales = list(categorias_ordinales.keys())

    orden_categorias = [
        categorias_ordinales[c]
        for c in cols_ordinales
    ]

    encoder = OrdinalEncoder(
        categories=orden_categorias
    )

    X[cols_ordinales] = encoder.fit_transform(
        X[cols_ordinales]
    )

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_selected = selector.fit_transform(X, y)

    scaler = MinMaxScaler()

    X_final = scaler.fit_transform(X_selected)

    return X_final, y
