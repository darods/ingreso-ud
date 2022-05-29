import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.write("""
# ¿Paso a la Distri?
Esta aplicación predice si puedes entrar a **ingeniería de sistemas** 
""")

st.header('Resultados del ICFES')


def user_input_features():
    p_icfes = st.number_input('Puntaje ICFES', 0, 500, 300)
    lenguaje = st.number_input('Lenguaje', 0, 100, 50)
    matematicas = st.number_input('Matematicas', 0, 100, 50)
    ciudadanas = st.number_input('Ciencias Ciudadanas', 0, 100, 50)
    naturales = st.number_input('Ciencias  Naturales', 0, 100, 50)
    ingles = st.number_input('Ingles', 0, 100, 50)
    '''
    matematicas 35%
    ciencias 35%
    lectura 15%
    sociales 10%
    ingles 5%
    '''
    ponderado = (matematicas*0.35)+(naturales*0.35)+(lenguaje*0.15)+(ciudadanas*0.1)+(ingles*0.05)
    
    data = {'PUNTAJE_ICFES': p_icfes,
            'PUNTAJE_PONDERADO': ponderado}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Reads in saved classification model
load_clf = pickle.load(open('ing_sistemas_model.pkl', 'rb'))


st.subheader('Datos ingresados por el usuario')
st.write(df)

st.subheader('Mapeo de resultado')
mapeo = {'ADMITIDO': 0, 'OPCIONADO': 2, 'NO': 1}
st.write(mapeo)

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Resultado de la predicción')
clasificacion = np.array(['ADMITIDO','NO','OPCIONADO'])
st.write(clasificacion[prediction])

st.subheader('Probabilidad de predicción')
st.write(prediction_proba)