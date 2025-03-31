import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Paso 8: Ejemplo de predicción para un nuevo candidato
nuevo_candidato = pd.DataFrame({
    'Años de Experiencia': [5],
    'Nivel Educativo': [le_educativo.transform(['Licenciatura'])[0]],  # Convertir el nivel educativo
    'Habilidad_C++': [0],  # Candidato sin habilidad en C++
    'Habilidad_Java': [0],  # Candidato sin habilidad en Java
    'Habilidad_JavaScript': [0],   # Candidato sin habilidad en Javascript
    'Habilidad_Python': [1]  # Candidato con habilidad en Python
   })  # Este es un ejemplo de características del nuevo candidato

prediccion = modelo.predict(nuevo_candidato)
habilidades = ""
if nuevo_candidato['Habilidad_C++'] [0] == 1:
    habilidades = "C++"
if nuevo_candidato['Habilidad_Java'] [0] == 1:
    habilidades = "Java"
if nuevo_candidato['Habilidad_JavaScript'] [0] == 1:
    habilidades = "JavaScript"
if nuevo_candidato['Habilidad_Python'] [0] == 1:
    habilidades = "Python"
    
    
print(f'El candidato  con {nuevo_candidato['Años de Experiencia'] [0]} años de experiencia y habilidad en {habilidades} es: {prediccion[0]}')
