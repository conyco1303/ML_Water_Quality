import streamlit as st
import pandas as pd
import pickle
from custom_functions import transform_data, modelos_ML

# Descripción de las clasificaciones del agua
descri_clasificacion = {
    'C1S1': 'Calidad de Agua C1S1: Las aguas bajas en salinidad y sodio son adecuadas para el riego y se pueden utilizar con la mayoría de los cultivos sin restricciones de uso en la mayoría de los suelos.',
    'C2S1': 'Calidad de Agua C2S1: Las aguas con salinidad media y bajo contenido de sodio son óptimas para el riego y se pueden utilizar en la mayoría de los suelos sin peligro de acumulación de sodio intercambiable. Los cultivos pueden cultivarse sin preocupaciones significativas de salinidad.',
    'C3S1': 'Calidad de Agua C3S1: Las aguas de alta salinidad y bajo contenido de sodio requieren un buen drenaje. Se deben seleccionar cultivos con buena tolerancia a la sal.',
    'C3S2': 'Calidad de Agua C3S2: Las aguas de alta salinidad y contenido medio de sodio requieren un buen drenaje y son adecuadas para suelos de textura gruesa u orgánicos con buena permeabilidad.',
    'C3S3': 'Calidad de Agua C3S3: Las aguas de alta salinidad y alto contenido de sodio requieren un manejo especial del suelo, drenaje eficiente, lixiviación abundante y adición de materia orgánica. El uso de enmiendas de yeso facilita su aprovechamiento.',
    'C4S1': 'Calidad de Agua C4S1: Las aguas de muy alta salinidad y bajo contenido de sodio no son aptas para el riego a menos que el suelo sea permeable y cuente con drenaje adecuado. Debe aplicarse un exceso de agua para lixiviar la sal. Se deben seleccionar cultivos altamente tolerantes a la sal.',
    'C4S2': 'Calidad de Agua C4S2: Las aguas de muy alta salinidad y contenido medio de sodio no son adecuadas para riego en suelos de textura fina y con baja lixiviación, pero pueden utilizarse en suelos de textura gruesa u orgánicos con buena permeabilidad.',
    'C4S3': 'Calidad de Agua C4S3: Las aguas con muy alta salinidad y alto contenido de sodio pueden causar niveles perjudiciales de sodio intercambiable en la mayoría de los suelos. Requieren un manejo especial del suelo, buen drenaje, alta lixiviación y adición de materia orgánica. La enmienda de yeso facilita su uso.',
    'C4S4': 'Calidad de Agua C4S4: Las aguas con muy alta salinidad y muy alto contenido de sodio generalmente no son aptas para el riego. Estas aguas contienen cloruro de sodio y pueden presentar riesgos de sodio. Pueden utilizarse en suelos de textura gruesa con excelente drenaje para cultivos extremadamente tolerantes a la sal. La enmienda de yeso es esencial para su aprovechamiento.'
}

# Cargar los modelos

with open('models/decision_tree_model.pkl','rb') as archivo:
    decision_tree_model=pickle.load(archivo)

with open('models/random_forest_model.pkl','rb') as archivo:
    random_forest_model=pickle.load(archivo)    
 
with open('models/knn_model.pkl','rb') as archivo:
    knn_model=pickle.load(archivo)  

with open('models/svm_model.pkl','rb') as archivo:
    svm_model=pickle.load(archivo)  


# Encabezado de la aplicación
st.title('Predicción de Calidad del Agua')

mean_values = {
    'gwl': 10.242452,
    'ec': 1336.079946,
    'tds': 855.091165, 
    'co3': 11.287418,
    'hco3': 295.692665,
    'cl': 188.590786,
    'f': 1.166641,
    'no3': 72.358543,
    'so4': 46.212398,
    'na': 123.504072,
    'k': 8.069593,
    'ph': 7.854038,
    # 'ca': 80.836314,
    # 'mg': 50.807989,
    # 'th': 410.868109,
    # 'sar': 2.795812,
}



# Creamos campos de entrada con valores iniciales basados en medias

gwl = st.number_input('Nivel de Agua Subterránea (gwl)', min_value=0.0, value=mean_values['gwl'], step=1.0)
ec = st.number_input('Conductividad Eléctrica (E.C)', min_value=0.0, value=mean_values['ec'], step=1.0)
tds = st.number_input('Total de Sólidos Disueltos (TDS)', min_value=0.0, value=mean_values['tds'], step=1.0)
co3 = st.number_input('Carbonatos (CO3)', min_value=0.0, value=mean_values['co3'], step=1.00)
hco3 = st.number_input('Bicarbonatos (HCO3)', min_value=0.0, value=mean_values['hco3'], step=1.0)
cl = st.number_input('Cloro (Cl)', min_value=0.0, value=mean_values['cl'], step=1.0)
f = st.number_input('Fluoruro (F)', min_value=0.0, value=mean_values['f'], step=1.00)
no3 = st.number_input('Nitrato (NO3)', min_value=0.0, value=mean_values['no3'], step=1.0)
so4 = st.number_input('Sulfato (SO4)', min_value=0.0, value=mean_values['so4'], step=1.0)
na = st.number_input('Sodio (Na)', min_value=0.0, value=mean_values['na'], step=1.0)
k = st.number_input('Potasio (K)', min_value=0.0, value=mean_values['k'], step=1.0)
ph = st.number_input('pH', min_value=0.0, value=mean_values['ph'], step=1.0)
# ca = st.number_input('Calcio (Ca)', min_value=0.0, value=mean_values['ca'], step=1.0)
# mg = st.number_input('Magnesio (Mg)', min_value=0.0, value=mean_values['mg'], step=1.0)
# th = st.number_input('Dureza Total (T.H)', min_value=0.0, value=mean_values['th'], step=1.0)
# sar = st.number_input('Relación de Adsorción de Sodio (SAR)', min_value=0.0, value=mean_values['sar'], step=1.0)


# Botón para seleccionar el modelo con el que quieres predecir
modelo_seleccionado = st.selectbox('Selecciona un modelo', ['Random Forest', 'Árbol de decisión', 'SVM', 'K-NN'])


# Cuando el usuario haga clic en un botón "Predecir", obtén los valores ingresados
if st.button('Predecir'):
    # Crea un DataFrame con los valores ingresados por el usuario
    input_data = pd.DataFrame({
        'gwl': [gwl],
        'ec': [ec],
        'tds': [tds],
        'co3': [co3],
        'hco3': [hco3],
        'cl': [cl],
        'f': [f],
        'no3': [no3],
        'so4': [so4],
        'na': [na],
        'k': [k],
        'ph': [ph]
        # 'ca': [ca],
        # 'mg': [mg],
        # 'th': [th],
        # 'sar': [sar],
    })

    # Predicciones según el modelo seleccionado
    if modelo_seleccionado == 'Random Forest':
        prediccion = random_forest_model.predict(input_data)
    elif modelo_seleccionado == 'Árbol de decisión':
        prediccion = decision_tree_model.predict(input_data)
    elif modelo_seleccionado == 'SVM':
        prediccion = svm_model.predict(input_data)
    elif modelo_seleccionado == 'K-NN':
        prediccion = knn_model.predict(input_data)

    # Obtener la descripción de la clasificación predicha
    descripcion_prediccion = descri_clasificacion.get(prediccion[0], 'Descripción no disponible')

    # Ver la predicción 
    st.write('La predicción es:', prediccion)
    st.write('Descripción: ', descripcion_prediccion)


