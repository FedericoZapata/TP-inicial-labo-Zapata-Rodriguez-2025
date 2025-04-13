from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Variables globales
modelo = None
le_educativo = LabelEncoder()
df = None

# Ruta principal: Página para cargar el archivo CSV
@app.route("/", methods=["GET", "POST"])
def index():
    global df, modelo, le_educativo
    if request.method == "POST":
        # Subir archivo CSV
        archivo = request.files["archivo"]
        if archivo:
            df = pd.read_csv(archivo)
            # Preprocesamiento
            le_educativo.fit(df['Nivel Educativo'])
            df['Nivel Educativo'] = le_educativo.transform(df['Nivel Educativo'])
            df = pd.concat([df, pd.get_dummies(df['Habilidades'], prefix='Habilidad')], axis=1)
            df.drop('Habilidades', axis=1, inplace=True)

            # Preparar datos para el modelo
            X = df.drop('Apto', axis=1)
            y = df['Apto']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Entrenar modelo
            modelo = DecisionTreeClassifier(random_state=42)
            modelo.fit(X_train, y_train)

            # Agregar predicciones al DataFrame
            df["Predicción"] = modelo.predict(X)

            return redirect(url_for("resultados"))
    return render_template("index.html")

# Ruta para mostrar los resultados
@app.route("/resultados")
def resultados():
    global df
    if df is not None:
        return render_template("resultados.html", datos=df.to_dict(orient="records"), columnas=df.columns)
    return redirect(url_for("index"))

# Ruta para predecir un nuevo candidato
@app.route("/predecir", methods=["GET", "POST"])
def predecir():
    global modelo, le_educativo
    if request.method == "POST":
        # Obtener datos del formulario
        años_experiencia = int(request.form["años_experiencia"])
        nivel_educativo = request.form["nivel_educativo"]
        habilidad = request.form["habilidad"]

        # Convertir datos al formato del modelo
        nivel_educativo_transformado = le_educativo.transform([nivel_educativo])[0]
        habilidades = {"Habilidad_C++": 0, "Habilidad_Java": 0, "Habilidad_JavaScript": 0, "Habilidad_Python": 0}
        habilidades[f"Habilidad_{habilidad}"] = 1

        # Crear DataFrame para predicción
        nuevo_candidato = pd.DataFrame({
            "Años de Experiencia": [años_experiencia],
            "Nivel Educativo": [nivel_educativo_transformado],
            **habilidades
        })

        # Realizar predicción
        prediccion = modelo.predict(nuevo_candidato)[0]
        return render_template("prediccion.html", prediccion=prediccion, candidato=nuevo_candidato.to_dict(orient="records")[0])
    return render_template("prediccion.html")

if __name__ == "__main__":
    app.run(debug=True)