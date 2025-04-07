# TP-inicial-labo-Zapata-Rodriguez-2025
## Primera entrega

## Instrucciones

#### Requisitos

- Python 3.x
- Librerias necesarias: 
```bash
pip install pandas sickit-learn
```
#### Archivos necesarios
- **Datos.csv:** archivo con el dataset

#### Como ejecutarlo

1. Abrir una terminal
2. Asegurarse de tener el archivos **Datos.csv** en la misma carpeta
3. Ejecutar el script: 

```bash
python clasificacion.py
```
#### Como entrenarlo

El codigo toma los datos del archivo **Datos.csv** y hace lo siguiente:

1. Preprocesamiento: 
    - Codifica la columna **Nivel Educativo** en valores numericos usando *LabelEncoder*
    - Convierte las habilidades en variables binarias con *get_dummies*

2. Separacion de datos:
    - Divide el dataset en 70% para entrenamiento y 30% para prueba, usando *train_test_split*
3. Entrenamiento:
    - Se entrena el modelo *DecisionTreeClassifier* con los datos de entrenamiento
4. Evaluacion
    - Se compara la predicción con los valores reales y se calcula la precisión.

#### Como hacer Predicciones?

Para hacer una predicción, se crea un nuevo DataFrame con los mismos campos que el resto del dataset: años de experiencia, nivel educativo (ya transformado en número), y las habilidades en formato binario (1 si tiene la habilidad, 0 si no).
Como por ejemplo:

```python
nuevo_candidato = pd.DataFrame({
    'Años de Experiencia': [5],
    'Nivel Educativo': [le_educativo.transform(['Licenciatura'])[0]],
    'Habilidad_C++': [0],
    'Habilidad_Java': [0],
    'Habilidad_JavaScript': [0],
    'Habilidad_Python': [1]
})
```

Luego se llama al método .predict() del modelo para obtener el resultado:

```python
modelo.predict(nuevo_candidato)
```

### Implementacion
Para la implementación preliminar del modelo de machine learning comenzamos creando un dataset ficticio en un archivo .csv que tiene  como columnas:

- Años de experiencia: un número entero
- Nivel educativo: que puede ser "Licenciatura", "Maestría" o "Doctorado"
- Habilidades: que puede ser  "Python", "Java", "C++", o "JavaScript"
- Apto: que indica si el candidato es “Apto” o “No apto

| años de experiencia | nivel educativo | Habilidades | Apto |
| ------------------- | -------------- | ------------ | ---- |
|5 | Licenciatura | Python | Apto |
|3 | Máster | Java | No apto | 
|8| Doctorado | JavaScript | Apto | 
|2| Licenciatura | C++ | No Apto| 
|10| Máster | Python | Apto | 

A continuación creamos un archivo .py para comenzar a implementar el modelo. Primero importamos las bibliotecas que vamos a utilizar: 

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_store
```

A continuación utilizamos la biblioteca Pandas para leer el archivo .csv con los datos

```python
#Paso 1: Leer el archivo CSV
df = pd.read_csv("Datos.csv")
```
Luego procedemos a  convertir las variables desde String a números enteros que el algoritmo pueda reconocer

```python
#Paso 2: Preprosecamiento de datos
#Convertir 'Nivel Educativo' en numeros
le_educativo = LabelEncoder()
df['Nivel Educativo'] = le_educativo.fit_transform(df['Nivel Educativo'])

#Convertir las habilidades en variables binarias
df = pd.concat([df, pd.get_dummies(df['Habilidades'], prefix='Habilidad')], axis=1)
df.drop = ('Habilidades',axis=1, inplace=True) 
```
A continuación procedemos a preparar los datos para el modelo: 
```python
#Paso 3: Preparar los datos para el modelo
X = df.drop('Apto', axis=1) #Caracteristicas (sin mostrar si es apto o no)
Y = df['Apto']

#Paso 4: Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, Y_train, y_train, x_train = train_test_split(X, y, test_size=0.3, random_state=42) #30% para la prueba
```

En X tenemos las variables que se utilizan para hacer las predicciones y en Y tenemos la variable objetivo(si el candidato es apto o no). Utilizando train_test_split podemos tomar algunos datos(un 70%) para realizar el entrenamiento del modelo y el resto(30%) para probarlo.
Una vez que tenemos los datos separados podemos utilizarlo con el modelo para entrenarlo y luego testearlo:

```python
#Paso 5: Crear y entrenar el modelo
modelo = DesicionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

#Paso 6: Hacer predicciones
y_pred = modelo.predict(X_test)
```
Con el modelo entrenado podemos evaluar su precisión utilizando Accuracy_score( en nuestro caso nos dio un 43.75%, que podría mejorarse utilizando una muestra de datos más grande)

```python
#Paso 7: evaluar el modelo
precision = accuracy_store(y_test, y_pred)
print(f'presicion del modelo: {precision * 100:.2f}%')
```
Finalmente podemos utilizar el modelo para predecir si un candidato es apto. 

```python
#Paso 8: Ejemplo de prediccion para un nuevo candidato
nuevo_candidato = pd.DataFrame((
    'Años de Experiencia': [5],
    'Nivel Educativo': [le_educativo.transform(['Licenciatura'])[0]], #convertir el nivel educativo
    'Habilidad_C++': [0], #Candidato sin habilidad en C++
    'Habilidad_Java': [0], #Candidato sin habilidad en Java
    'Habilidad_JavaScript':[0], #Candidato sin habilidad en JS
    'Habilidad_Python': [1], #Candidato con habilidad en Pyhton
)) #Este es un ejemplo de caracteristicas de nuevo candidato

prediccion = modelo.predict(nuevo_candidato)
habilidades = ""
if nuevo_candidato['Habilidad_C++'] [0] == 1
    habilidades = "C++"
if nuevo_candidato['Habilidad_Java'] [0] == 1
    habilidades = "Java"
if nuevo_candidato['Habilidad_JavaScript'] [0] == 1
    habilidades = "JavaScript"
if nuevo_candidato['Habilidad_Python'] [0] == 1
    habilidades = "Python"

print(f'El candidato con {nuevo_candidato['Años de experiencia'][0]} años de experiencia y habilidad en {habilidades} es : {prediccion[0]}')
```