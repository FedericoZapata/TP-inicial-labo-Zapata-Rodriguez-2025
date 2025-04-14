from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)


modelo = None
le_educativo = LabelEncoder()
df = None
precision_modelo = None  # Nueva variable para almacenar la precisión del modelo

# Función para cargar y entrenar el modelo al iniciar la aplicación
def inicializar_modelo():
    global modelo, le_educativo, df, precision_modelo
    try:
        # Cargar los datos desde Datos.csv
        df = pd.read_csv("Datos.csv")

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

        # Calcular precisión del modelo
        precision_modelo = modelo.score(X_test, y_test) * 100  # Precisión en porcentaje

        print(f"Modelo entrenado correctamente con una precisión del {precision_modelo:.2f}%.")
    except Exception as e:
        print(f"Error al inicializar el modelo: {e}")

# Ruta principal: Página para cargar un archivo CSV
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
    global modelo, le_educativo, precision_modelo
    if request.method == "POST":
        # Subir archivo CSV con candidatos para predicción
        archivo = request.files.get("archivo")  # Obtener el archivo del formulario
        if archivo:
            # Leer el archivo CSV
            df_candidatos = pd.read_csv(archivo)

            # Preprocesar los datos del archivo
            df_candidatos['Nivel Educativo'] = df_candidatos['Nivel Educativo'].str.lower()  # Convertir a minúsculas
            df_candidatos['Nivel Educativo'] = le_educativo.transform(df_candidatos['Nivel Educativo'])
            df_candidatos = pd.concat([df_candidatos, pd.get_dummies(df_candidatos['Habilidades'], prefix='Habilidad')], axis=1)
            df_candidatos.drop('Habilidades', axis=1, inplace=True)

            # Asegurarse de que las columnas coincidan con las del modelo
            columnas_modelo = modelo.feature_names_in_
            for col in columnas_modelo:
                if col not in df_candidatos:
                    df_candidatos[col] = 0  # Agregar columnas faltantes con valor 0

            # Realizar predicciones
            X_candidatos = df_candidatos[columnas_modelo]
            df_candidatos['Predicción'] = modelo.predict(X_candidatos)

            # Convertir los valores de 'Nivel Educativo' a texto
            df_candidatos['Nivel Educativo'] = le_educativo.inverse_transform(df_candidatos['Nivel Educativo'])

            # Convertir valores booleanos o binarios a texto (ejemplo: experiencia)
            for col in df_candidatos.columns:
                if col.startswith("Habilidad_"):  # Convertir habilidades a "Sí" o "No"
                    df_candidatos[col] = df_candidatos[col].apply(lambda x: "Sí" if x == 1 else "No")

            # Convertir los resultados a un formato para la tabla
            datos = df_candidatos.to_dict(orient="records")
            columnas = df_candidatos.columns

            # Renderizar la plantilla con los resultados y la precisión del modelo
            return render_template(
                "prediccion.html",
                datos=datos,
                columnas=columnas,
                precision=precision_modelo
            )
    
    # Si es un GET, renderizar la página sin datos
    return render_template("prediccion.html", datos=None, columnas=None, precision=None)
# Inicializar el modelo al iniciar la aplicación
inicializar_modelo()

if __name__ == "__main__":
    app.run(debug=True)