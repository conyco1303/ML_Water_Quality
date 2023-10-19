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
    data_agua = pd.concat([data_2018, data_2019, data_2020])
    
    col_eliminar= ['sno', 'district', 'mandal', 'village', 'lat_gis', 'long_gis','RSC  meq  / L','Classification.1', 'anyo']
    data_agua=data_agua.drop(columns=col_eliminar)

    # LAs clasificaciones OG, O.G, C3S4 y C2S2 solo continene 1 registro cada una. Causan demasiado ruido, por tanto las eliminamos  
    data_agua = data_agua[~data_agua['Classification'].isin(['C3S4', 'C2S2','OG','O.G'])]
    
    # Calcular la media de la columna 'CO3'
    mean_co3 = data_agua['CO3'].mean()
    
    # Sustituir los valores nulos en 'CO3' con la media de la columna 'CO3'
    data_agua['CO3'].fillna(mean_co3, inplace=True)
    
    # Sustituir los valores nulos en 'gwl' con la media de la columna 'gwl'
    data_agua['gwl'].fillna(data_agua['gwl'].mean(), inplace=True)
    
    # # Sustituir los valores nulos en 'gwl' de data_2020 con la media de la columna 'gwl'
    # data_2020['gwl'].fillna(data_2020['gwl'].mean(), inplace=True)
    
    return data_2018, data_2019, data_2020, data_agua

# Llama a la función con tus DataFrames
data_2018, data_2019, data_2020, data_agua = transform_data(data_2018, data_2019, data_2020)


def modelos_ML(data_agua):
    X = data_agua.drop(['Classification'], axis=1)
    y = data_agua['Classification']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Normalizacion de los datos 
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=12)
    X_train_pca = pca.fit_transform(X_train_normalized)
    X_test_pca = pca.transform(X_test_normalized)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train_pca, y_train)
    dt_predictions = dt_classifier.predict(X_test_pca)
    dt_accuracy = accuracy_score(y_test, dt_predictions)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train_pca, y_train)
    rf_predictions = rf_classifier.predict(X_test_pca)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    # Support Vector Classifier
    svc_classifier = SVC()
    svc_classifier.fit(X_train_pca, y_train)
    svc_predictions = svc_classifier.predict(X_test_pca)
    svc_accuracy = accuracy_score(y_test, svc_predictions)

    # K-Nearest Neighbors
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train_pca, y_train)
    knn_predictions = knn_classifier.predict(X_test_pca)
    knn_accuracy = accuracy_score(y_test, knn_predictions)

     # Guarda los modelos en archivos .pkl
    with open('models/decision_tree_model.pkl', 'wb') as file:
        pickle.dump(dt_classifier, file)

    with open('models/random_forest_model.pkl', 'wb') as file:
        pickle.dump(rf_classifier, file)

    with open('models/svm_model.pkl', 'wb') as file:
        pickle.dump(svc_classifier, file)

    with open('models/knn_model.pkl', 'wb') as file:
        pickle.dump(knn_classifier, file)

    return {
        "explained_variance_ratio": explained_variance_ratio,
        "decision_tree_accuracy": dt_accuracy,
        "random_forest_accuracy": rf_accuracy,
        "svc_accuracy": svc_accuracy,
        "knn_accuracy": knn_accuracy,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,  # Agregar X_test_pca aquí
        "dt_predictions": dt_predictions
    }
    
   
# Llama a la función con tu conjunto de datos Water_data
resultados = modelos_ML(data_agua)
print("Explained Variance Ratio:", resultados["explained_variance_ratio"])
print("Decision Tree Accuracy:", resultados["decision_tree_accuracy"])
print("Random Forest Accuracy:", resultados["random_forest_accuracy"])
print("SVC Accuracy:", resultados["svc_accuracy"])
print("K-NN Accuracy:", resultados["knn_accuracy"])