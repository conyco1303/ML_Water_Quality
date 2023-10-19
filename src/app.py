import streamlit as st
import pandas as pd
import pickle

# Cargar los modelos
def cargar_modelos():
    decision_tree_model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
    random_forest_model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
    svc_model = pickle.load(open('models/svm_model.pkl', 'rb'))
    knn_model = pickle.load(open('models/knn_model.pkl', 'rb'))
    
    return decision_tree_model, random_forest_model, svc_model, knn_model

# Función para realizar predicciones con los modelos
def predecir_con_modelos(modelos, datos_entrada):
    decision_tree_model, random_forest_model, svc_model, knn_model = modelos

    # Predecimos con los cuatro modelos
    predicciones_decision_tree = decision_tree_model.predict(datos_entrada)
    predicciones_random_forest = random_forest_model.predict(datos_entrada)
    predicciones_svc = svc_model.predict(datos_entrada)
    predicciones_knn = knn_model.predict(datos_entrada)

    return {
        "Decision Tree Predictions": predicciones_decision_tree,
        "Random Forest Predictions": predicciones_random_forest,
        "SVC Predictions": predicciones_svc,
        "K-NN Predictions": predicciones_knn
    }

# Cargar modelos
modelos_cargados = cargar_modelos()

# Encabezado de la aplicación
st.title('Predicción de Calidad del Agua')

# Crear formularios para que los usuarios ingresen valores
st.header('Ingrese los valores de las 12 variables:')
valores_usuario = {}
for columna in data_agua.drop(['Classification'], axis=1).columns:
    valores_usuario[columna] = st.number_input(columna, min_value=0.0)

# Botón para realizar predicciones
if st.button('Realizar Predicciones'):
    # Crear un DataFrame con los valores ingresados por el usuario
    datos_usuario = pd.DataFrame([valores_usuario])

    # Realizar predicciones con los modelos
    predicciones = predecir_con_modelos(modelos_cargados, datos_usuario)

    # Mostrar las predicciones
    st.subheader('Predicciones:')
    for modelo, predicciones_modelo in predicciones.items():
        st.write(f'{modelo}: {predicciones_modelo[0]}')



