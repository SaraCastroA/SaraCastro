import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer

def generar_caso_de_uso_vectorizar_sentencias():
    n_rows = random.randint(4, 7)

    sujetos = ["juez", "tribunal", "fiscal", "defensa", "acusado", "demandante"]
    acciones = ["ordena", "rechaza", "aprueba", "suspende", "admite", "resuelve"]
    objetos = ["demanda", "recurso", "prueba", "audiencia", "sentencia", "apelacion"]

    textos = [
        f"{random.choice(sujetos)} {random.choice(acciones)} {random.choice(objetos)}"
        for _ in range(n_rows)
    ]

    df = pd.DataFrame({"texto": textos})

    input_data = {"df": df.copy()}

    vectorizer = TfidfVectorizer()
    X_expected = vectorizer.fit_transform(df["texto"]).toarray()
    palabras_expected = list(vectorizer.get_feature_names_out())

    output_data = (X_expected, palabras_expected)

    print("=== INPUT GENERADO (Sentencias Legales) ===")
    print(df)

    print("\n=== OUTPUT ESPERADO ===")
    print("Matriz TF-IDF:")
    print(X_expected)
    print("\nPalabras:")
    print(palabras_expected)

    return input_data, output_data
