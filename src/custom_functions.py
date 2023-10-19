import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def transform_data(data_2018, data_2019, data_2020):
    # Eliminar columnas "season" de los DataFrames
    for df in [data_2018, data_2019,data_2020]:
        if 'season' in df.columns:
            df.drop("season", axis=1, inplace=True)
    
    # Eliminar la columna "Unnamed: 8" del DataFrame de 2020
    if 'Unnamed: 8' in data_2020.columns:
        data_2020.drop("Unnamed: 8", axis=1, inplace=True)
    
    # Agregar la columna 'anyo' a cada DataFrame
    data_2018['anyo'] = 2018
    data_2019['anyo'] = 2019
    data_2020['anyo'] = 2020
    
    # Renombrar columnas según el diccionario
    column_name = {'CO_-2 ': 'CO3', 'HCO_ - ': 'HCO3', 'Cl -': 'Cl', 'F -': 'F', 'NO3- ': 'NO3 ', 
                   'SO4-2': 'SO4', 'Na+': 'Na', 'K+': 'K', 'Ca+2': 'Ca', 'Mg+2': 'Mg', 'EC': 'E.C'}
    data_2019.rename(columns=column_name, inplace=True)
    
    # Actualizar un valor específico en data_2020
    data_2020.at[261, 'pH'] = 8.05
    
    # Calcular las medias de la columna "CO3" por distrito en 2018, 2019 y 2020
    mean_co3_2018 = data_2018.groupby('district')['CO3'].mean()
    mean_co3_2019 = data_2019.groupby('district')['CO3'].mean()
    mean_co3_2020 = data_2020.groupby('district')['CO3'].mean()
    
    # Crear un diccionario para mapear las medias por distrito en 2019
    co3_mean_dict_2019 = mean_co3_2019.to_dict()
    
    # Sustituir los valores nulos en 2019 con las medias por distrito de 2019
    data_2019['CO3'].fillna(data_2019['district'].map(co3_mean_dict_2019), inplace=True)
    
    # Concatenar los DataFrames data_2018 y data_2019 en data_agua
    data_agua = pd.concat([data_2018, data_2019])
    
    # Calcular la media de la columna 'CO3'
    mean_co3 = data_agua['CO3'].mean()
    
    # Sustituir los valores nulos en 'CO3' con la media de la columna 'CO3'
    data_agua['CO3'].fillna(mean_co3, inplace=True)
    
    # Sustituir los valores nulos en 'gwl' con la media de la columna 'gwl'
    data_agua['gwl'].fillna(data_agua['gwl'].mean(), inplace=True)
    
    # Sustituir los valores nulos en 'gwl' de data_2020 con la media de la columna 'gwl'
    data_2020['gwl'].fillna(data_2020['gwl'].mean(), inplace=True)
    
    return data_2018, data_2019, data_2020, data_agua


def modelos_ML(data_agua, data_2020):
    # Dividir los datos en conjuntos de entrenamiento (2018 y 2019) y prueba (2020)
    X_train = data_agua.drop(columns=['Classification', 'Classification.1'])
    y_train = data_agua['Classification']
    X_test = data_2020.drop(columns=['Classification', 'Classification.1'])
    y_test = data_2020['Classification']

    # Seleccionar las características numéricas y categóricas
    numeric_features = X_train.select_dtypes(include=['float64', 'int64'])
    categorical_features = X_train.select_dtypes(include=['object'])

    # Crear transformadores para las características numéricas y categóricas
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combinar los transformadores utilizando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features.columns),
            ('cat', categorical_transformer, categorical_features.columns)
        ])

    # Crear un pipeline que incluye la transformación y el modelo (puedes reemplazar LinearRegression con el modelo que prefieras)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestClassifier())  # Puedes reemplazar esto con el modelo que estás utilizando
    ])

    # Aplicar la transformación a los datos de entrenamiento
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Aplicar la transformación a los datos de prueba
    X_test_scaled = preprocessor.transform(X_test)

    # Entrenar los modelos
    dt_classifier = DecisionTreeClassifier()
    rf_classifier = RandomForestClassifier()
    svm_classifier = SVC()

    dt_classifier.fit(X_train_scaled, y_train)
    rf_classifier.fit(X_train_scaled, y_train)
    svm_classifier.fit(X_train_scaled, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred_dt = dt_classifier.predict(X_test_scaled)
    y_pred_rf = rf_classifier.predict(X_test_scaled)
    y_pred_svm = svm_classifier.predict(X_test_scaled)

    # Evaluar los modelos y comparar su rendimiento
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    # Imprimir informe de clasificación para cada modelo
    print("Informe de clasificación del Árbol de Decisión:")
    print(classification_report(y_test, y_pred_dt))
    
    print("Informe de clasificación del Bosque Aleatorio:")
    print(classification_report(y_test, y_pred_rf))
    
    print("Informe de clasificación de la Máquina de Vectores de Soporte (SVM):")
    print(classification_report(y_test, y_pred_svm))
    
    return {
        'dt_classifier': dt_classifier,
        'rf_classifier': rf_classifier,
        'svm_classifier': svm_classifier,
        'accuracy_dt': accuracy_dt,
        'accuracy_rf': accuracy_rf,
        'accuracy_svm': accuracy_svm,
        'y_pred_dt': y_pred_dt,
        'y_pred_rf': y_pred_rf,
        'y_pred_svm': y_pred_svm,
        'y_test': y_test  # Devolver y_test como parte de los resultados
    }