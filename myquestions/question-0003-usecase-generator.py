import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def generar_caso_de_uso_agrupar_jugadores():
    n_rows = random.randint(6, 10)
    n_clusters = random.randint(2, 4)

    df = pd.DataFrame({
        "goles": np.random.randint(0, 20, size=n_rows),
        "asistencias": np.random.randint(0, 15, size=n_rows),
        "pases_clave": np.random.randint(5, 50, size=n_rows),
        "recuperaciones": np.random.randint(1, 40, size=n_rows)
    })

    input_data = {
        "df": df.copy(),
        "n_clusters": n_clusters
    }

    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    output_data = modelo.fit_predict(X_scaled)

    print("=== INPUT GENERADO (Estadísticas de Jugadores) ===")
    print(df)
    print("\nNúmero de clusters solicitado:", n_clusters)

    print("\n=== OUTPUT ESPERADO (Etiquetas de Clusters) ===")
    print(output_data)

    return input_data, output_data


if __name__ == "__main__":
    generar_caso_de_uso_agrupar_jugadores()
