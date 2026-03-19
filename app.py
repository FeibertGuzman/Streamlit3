import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Pokémon Evolution & Type Prediction Project")
st.write("Análisis de datos de Pokémon y predicción de tipo dual.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("text2.csv")
        return df
    except FileNotFoundError:
        st.error("Archivo text2.csv no encontrado.")
        return None

df = load_data()

if df is not None:
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df.head(10))

    # Configuración en la barra lateral
    st.sidebar.header("Configuración del Modelo")
    epochs = st.sidebar.slider("Épocas de Entrenamiento (Red Neuronal)", 10, 500, 150)
    st.sidebar.info("Objetivo: Predecir 'is_dual_type'")

    if st.sidebar.button("Entrenar Modelos"):
        # Preprocesamiento
        df['is_dual_type'] = df['Type2'].notnull().astype(1)
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X = encoder.fit_transform(df[['Type1']])
        y = df['is_dual_type']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelos
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
        lr = LinearRegression()

        mlp.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        # Métricas
        models = ['Red Neuronal (MLP)', 'Regresión Lineal']
        mse = [mean_squared_error(y_test, mlp.predict(X_test)), mean_squared_error(y_test, lr.predict(X_test))]
        r2 = [r2_score(y_test, mlp.predict(X_test)), r2_score(y_test, lr.predict(X_test))]

        results = pd.DataFrame({'Modelo': models, 'MSE': mse, 'R2 Score': r2})
        
        st.subheader("Resultados del Entrenamiento")
        st.table(results)

        # Gráficos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(models, mse, color=['#3b82f6', '#fb923c'])
        ax1.set_title("MSE (Menor es mejor)")
        
        ax2.bar(models, r2, color=['#3b82f6', '#fb923c'])
        ax2.set_title("R2 Score (Mayor es mejor)")
        
        st.pyplot(fig)
        st.success(f"¡Modelos entrenados exitosamente con {epochs} épocas!")
else:
    st.info("Esperando la carga del dataset para continuar...")
