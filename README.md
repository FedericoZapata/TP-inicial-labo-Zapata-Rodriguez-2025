# TP-inicial-labo-Zapata-Rodriguez-2025


### Instrucciones

#### Requisitos 📚

- Python 3.x
- Librerias necesarias: 
```bash
pip install pandas sickit-learn
pip install flask
```

En este caso utilizamos flask para la interfaz visual de nuestra aplicacion.

#### Archivos necesarios
- **Datos.csv:** archivo con el dataset
- **Candidatos.csv** archivo con el que cargamos los candidatos a evaluar

#### Como ejecutarlo

1. Abrir una terminal
2. Asegurarse de tener el archivos **Datos.csv** en la misma carpeta
3. Asegurarse de tener los datos necesarios para cargar en la pagina, en este caso **Candidatos.csv**
4. Ejecutar el script: 

```bash
flask run
```
### Cómo funciona el modelo

1. **Entrenamiento:**
   - El modelo se entrena utilizando los datos del archivo `Datos.csv`.
   - Se realiza un preprocesamiento para convertir las columnas en un formato que el modelo pueda interpretar:
     - `Nivel Educativo` se codifica como valores numéricos usando `LabelEncoder`.
     - `Habilidades` se convierten en variables binarias usando `get_dummies`.

2. **Predicción:**
   - El modelo utiliza un árbol de decisión (`DecisionTreeClassifier`) para clasificar a los candidatos como `Apto` o `No Apto`.
   - La predicción se basa en las características del candidato, como `Años de Experiencia`, `Nivel Educativo` y `Habilidades`.

3. **Visualización:**
   - Los resultados se muestran en una tabla con los datos originales y la predicción.
   - Los valores se convierten a un formato legible:
     - `Nivel Educativo`: Se muestra como `autodidacta`, `tecnicatura` o `licenciatura`.
     - `Habilidades`: Se muestran como `Sí` o `No`.

---

### Cómo hacer predicciones

#### 1. Preparar el archivo CSV
El archivo debe contener las siguientes columnas:
- **Años de Experiencia:** Número entero que indica los años de experiencia del candidato.
- **Nivel Educativo:** Texto que puede ser `autodidacta`, `tecnicatura` o `licenciatura`.
- **Habilidades:** Texto que indica una habilidad específica, como `Python`, `Java`, `C++` o `JavaScript`.

Ejemplo de archivo `Candidatos.csv`:
```csv
Años de Experiencia,Nivel Educativo,Habilidades
5,Licenciatura,Python
3,Tecnicatura,Java
7,Autodidacta,C++
10,Licenciatura,JavaScript
2,Tecnicatura,Python
```

Luego se llama al método .predict() del modelo para obtener el resultado:

```python
modelo.predict(nuevo_candidato)
```

#### 2. Subir el archivo CSV
- Accede a la página principal de la aplicación.
- Usa el formulario para seleccionar y cargar el archivo `Candidatos.csv`.

#### 3. Procesamiento y predicción
- La aplicación procesará automáticamente los datos del archivo:
  - Convertirá el nivel educativo (`autodidacta`, `tecnicatura`, `licenciatura`) en un formato que el modelo pueda interpretar.
  - Transformará las habilidades en variables binarias (`Sí` o `No`).
- El modelo realizará las predicciones para cada candidato.

#### 4. Visualización de resultados
- Los resultados se mostrarán en una tabla en la página de resultados.
- La tabla incluirá:
  - Los datos originales del candidato.
  - La predicción (`Apto` o `No Apto`).
- Además, se resaltarán los candidatos aptos en verde y los no aptos en rojo para facilitar la visualización.

---

### Resultados del modelo

**Precisión del modelo:** 93.33%

**Distribución de clases:**

| Clase     | Cantidad |
|-----------|----------|
| Apto      | 64       |
| No apto   | 34       |

**Matriz de confusión:**

|                      | Predicho Apto | Predicho No Apto |
|----------------------|---------------|------------------|
| **Realmente Apto**   | 16 (✔️)        | 0 (❌)            |
| **Realmente No Apto**| 2 (❌)         | 12 (✔️)           |

- **Verdaderos positivos (TP):** 16 → Aptos bien clasificados.
- **Falsos positivos (FP):** 2 → No aptos mal clasificados como aptos.
- **Verdaderos negativos (TN):** 12 → No aptos correctamente detectados.
- **Falsos negativos (FN):** 0 → Aptos mal clasificados como no aptos.

**Reporte de clasificación:**

| Clase     | Precisión | Recall | F1-score | Soporte |
|-----------|-----------|--------|----------|---------|
| Apto      | 0.89      | 1.00   | 0.94     | 16      |
| No Apto   | 1.00      | 0.86   | 0.92     | 14      |

- **Accuracy (precisión total):** 93%
- **Macro promedio:** Precisión = 0.94, Recall = 0.93, F1 = 0.93.
- **Promedio ponderado:** Precisión = 0.94, Recall = 0.93, F1 = 0.93.

---

## Implementacion
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
from flask import Flask, render_template, request, redirect, url_for
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
    print("1. Licenciatura")
    print("2. Máster")
    print("3. Doctorado")
    nivel_educativo = input("Ingresa la opcion correspondiente al nivel educativo (1-3): ")
    if nivel_educativo not in ['1', '2', '3']:
        print("Por favor ingresa un número válido entre 1 y 3.")

# conversion del nivel educativo para que lo comprenda el sistema
nivel_educativo = int(nivel_educativo)
nivel_educativo_transformado = le_educativo.transform(['Licenciatura' if nivel_educativo == 1 else 'Máster' if nivel_educativo == 2 else 'Doctorado'])[0]

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


```

### Resultados del modelo

**Precisión del modelo:** 93.33%

**Distribucion de clases:**

| Clase     | Cantidad |
|-----------|----------|
| Apto      | 64       |
| No apto   | 34       |

**Matriz de confusion**

|                      | Predicho Apto | Predicho No Apto |
|----------------------|---------------|------------------|
| **Realmente Apto**   | 16 (✔️)        | 0 (❌)            |
| **Realmente No Apto**| 2 (❌)         | 12 (✔️)           |

- **Verdaderos positivos (TP):** 16 → Aptos bien clasificados
- **Falsos positivos (FP):** 2 → No aptos mal clasificados como aptos
- **Verdaderos negativos (TN):** 12 → No aptos correctamente detectados
- **Falsos negativos (FN):** 0 → Aptos mal clasificados como no aptos

**Reporte de clasificacion**

| Clase     | Precisión | Recall | F1-score | Soporte |
|-----------|-----------|--------|----------|---------|
| Apto      | 0.89      | 1.00   | 0.94     | 16      |
| No Apto   | 1.00      | 0.86   | 0.92     | 14      |

- **Accuracy (precisión total):** 93%
- **Macro promedio:** Precision = 0.94, Recall = 0.93, F1 = 0.93
- **Promedio ponderado:** Precision = 0.94, Recall = 0.93, F1 = 0.93

**Interpretacion**
- El modelo **identifica correctamente a todos los candidatos aptos** (recall 1.00 para clase “apto”).
- Tiene una **alta precisión al descartar candidatos no aptos** (precision 1.00 para clase “no apto”).
- Comete pocos errores al clasificar personas no aptas como aptas (2 casos).



### Capturas
![imagen 3](https://github.com/user-attachments/assets/947e8d50-49d4-4136-842b-a7d93947ad4a)
![imagen 2](https://github.com/user-attachments/assets/7da3433e-ea3d-4c70-af24-1912142b824f)
![imagen 1](https://github.com/user-attachments/assets/4116eff7-150c-4e00-a4b4-592c16c1d7bc)


