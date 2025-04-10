import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paso 1: Leer el archivo CSV
df = pd.read_csv("Datos.csv")

# Mostrar las primeras filas para ver cómo es el dataset
print("Datos importados:")
print(df.head())

# Paso 2: Preprocesamiento de datos
# Convertir 'Nivel Educativo' en números
le_educativo = LabelEncoder()
df['Nivel Educativo'] = le_educativo.fit_transform(df['Nivel Educativo'])

# Convertir las habilidades en variables binarias
df = pd.concat([df, pd.get_dummies(df['Habilidades'], prefix='Habilidad')], axis=1)
df.drop('Habilidades', axis=1, inplace=True)

# Paso 3: Preparar los datos para el modelo
X = df.drop('Apto', axis=1)  # Características (sin mostrar si es apto o no)
y = df['Apto']  

# Paso 4: Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 30% para prueba

# Paso 5: Crear y entrenar el modelo
modelo = DecisionTreeClassifier(random_state=42)
#modelo = LogisticRegression( random_state=42)
modelo.fit(X_train, y_train)
print(f"Entrenando modelo con los datos ingresados")

# Paso 6: Hacer predicciones
y_pred = modelo.predict(X_test)
print(f"Testeando modelo")

# Paso 7: Evaluar el modelo
precision = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {precision * 100:.2f}%')
print(df['Apto'].value_counts())

print("Matriz de confusión:")#muestra la cantidad de de falsos positivos y falsos negativos
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")#muestra la precision y sensibilidad
print(classification_report(y_test, y_pred))

# Paso 8: solicita datos por consola del candidato a testear
#años de experiencia
while True:
    try:
        años_experiencia = int(input("Ingrese los años de experiencia del candidato: "))
        if años_experiencia < 0:
            print("Por favor ingrese un número valido para los años de experiencia.")
        else:
            break
    except ValueError:
        print("Por favor ingresa un numero válido para los años de experiencia.")

#nivel educativo
nivel_educativo = ""
while nivel_educativo not in ['1', '2', '3']:
    print("Selecciona el nivel educativo del candidato:")
    print("1. Autodidacta")
    print("2. Tecnicatura")
    print("3. Licenciatura")
    nivel_educativo = input("Ingresa la opcion correspondiente al nivel educativo (1-3): ")
    if nivel_educativo not in ['1', '2', '3']:
        print("Por favor ingresa un número válido entre 1 y 3.")

# conversion del nivel educativo para que lo comprenda el sistema
nivel_educativo = int(nivel_educativo)
nivel_educativo_transformado = le_educativo.transform(['autodidacta' if nivel_educativo == 1 else 'tecnicatura' if nivel_educativo == 2 else 'licenciatura'])[0]

# habilidades, solo se permite seleccionar una
habilidad_seleccionada = 0
while habilidad_seleccionada not in [1, 2, 3, 4]:
    print("Selecciona la habilidad que tiene el candidato:")
    print("1. C++")
    print("2. Java")
    print("3. JavaScript")
    print("4. Python")
    try:
        habilidad_seleccionada = int(input("Ingresa el número de la habilidad (1-4): "))
        if habilidad_seleccionada not in [1, 2, 3, 4]:
            print("Selección no válida, por favor elige una opción entre 1 y 4.")
    except ValueError:
        print("Por favor ingresa un número válido entre 1 y 4.")

#conversion de las habilidades para que las comprenda el sistema

habilidades = ""
if habilidad_seleccionada == 1:
    habilidad_cpp = 1
    habilidad_java = 0
    habilidad_javascript = 0
    habilidad_python = 0
    habilidades = "C++"
elif habilidad_seleccionada == 2:
    habilidad_cpp = 0
    habilidad_java = 1
    habilidad_javascript = 0
    habilidad_python = 0
    habilidades = "Java"
elif habilidad_seleccionada == 3:
    habilidad_cpp = 0
    habilidad_java = 0
    habilidad_javascript = 1
    habilidad_python = 0
    habilidades = "JavaScript"
elif habilidad_seleccionada == 4:
    habilidad_cpp = 0
    habilidad_java = 0
    habilidad_javascript = 0
    habilidad_python = 1
    habilidades = "Python"

# Crear el DataFrame con los datos ingresados
nuevo_candidato = pd.DataFrame({
    'Años de Experiencia': [años_experiencia],
    'Nivel Educativo': [nivel_educativo_transformado],
    'Habilidad_C++': [habilidad_cpp],
    'Habilidad_Java': [habilidad_java],
    'Habilidad_JavaScript': [habilidad_javascript],
    'Habilidad_Python': [habilidad_python]
})

prediccion = modelo.predict(nuevo_candidato)

print(f'El candidato con {nuevo_candidato["Años de Experiencia"] [0]} años de experiencia y habilidad en {habilidades} es: {prediccion[0]}')
