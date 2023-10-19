Proyecto de Análisis y Predicción del Agua

Este proyecto se enfoca en el análisis y predicción de la calidad del agua en función de varios parámetros y características geográficas. Utiliza Machine Learning para clasificar la calidad del agua en diferentes distritos y años.

Bibliotecas:

Este proyecto utiliza varias bibliotecas de Python para realizar análisis de datos y construir modelos de Machine Learning, entre ellas:

NumPy
Pandas
Seaborn
Matplotlib
Streamlit
Scikit-Learn


Transformación de Datos:

Para preparar los datos antes de la construcción de modelos, se realiza una serie de transformaciones, como:

* Eliminación de columnas innecesarias.
* Agregar el año a cada conjunto de datos.
* Renombrar columnas según un diccionario.
* Manejo de valores nulos y sustitución con medias por distrito.
* Concatenación de los conjuntos de datos de diferentes años.
* Modelos de Machine Learning

El proyecto construye modelos de Machine Learning para predecir la calidad del agua. Estos modelos incluyen:

* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (K-NN)
* Preprocesamiento de Datos
* El preprocesamiento de datos incluye normalización y reducción de dimensionalidad mediante PCA.

Entrenamiento y Evaluación:

Los modelos se entrenan y evalúan utilizando datos de entrenamiento y prueba. La métrica de evaluación principal es la precisión (accuracy).

Guardado de Modelos:

Los modelos entrenados se guardan en archivos .pkl para su uso posterior.

Predicciones con Modelos:

El proyecto permite cargar modelos entrenados y realizar predicciones con ellos. Puedes utilizar estos modelos para predecir la calidad del agua en nuevos datos de entrada.

Importancia de Características:

Se realiza un análisis para determinar la importancia de las características utilizadas en el modelo Random Forest. Las características más importantes se muestran en orden descendente.

Uso: 

Para utilizar este proyecto, sigue los pasos a continuación:

* Instala las bibliotecas requeridas utilizando el comando: pip install -r requirements.txt.
* Realiza la transformación de datos ejecutando transform_data(data_2018, data_2019, data_2020).
* Entrena y evalúa los modelos con modelos_ML(data_agua).
* Guarda y carga modelos entrenados con cargar_modelos() y predecir_con_modelos(modelos, datos_entrada).
* Explora la importancia de características utilizando el código proporcionado.


Autor: Cono De Paola Prato 

Agradecimientos:

Agradezco a todas las personas y recursos por parte de The Bridge para conseguir este proyecto. En especial a mi familia por aguantarme y alimentarme mientras pasaba horas en el ordenador. A mis profesores Marco, Daniel y al más increible TA Ramón (No te vayas de ningún bottcamp sin preguntarle donde puedes comprar camisetas de fútbol)

