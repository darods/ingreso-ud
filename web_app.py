import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
st.set_page_config(layout="wide")

# get images
image_happy = Image.open('img/happy_pepe.png')
image_confused = Image.open('img/confused_pepe.png')
image_sad = Image.open('img/sad_pepe.png')

st.title('¿Paso a la Distri?')
st.write('Aplicación **NO OFICIAL** que predice tus chances de entrar a alguna de las carreras de la **Facultad de ingeniería (sede la 40)**')

# definir carreras
carrera = st.selectbox('Seleccione la carrera', ('Sistemas', 'Electrónica', 'Eléctrica', 'Industrial','Catastral'))

col1, col2 = st.columns((4, 2))




matematicas = st.sidebar.number_input('Matematicas', 0, 100, 75)
naturales = st.sidebar.number_input('Ciencias  Naturales', 0, 100, 75)
lenguaje = st.sidebar.number_input('Lenguaje', 0, 100, 70)
ciudadanas = st.sidebar.number_input('Ciencias Ciudadanas', 0, 100, 70)
ingles = st.sidebar.number_input('Ingles', 0, 100, 70)

def input_ingenieria_general():

    p_icfes = matematicas + naturales + lenguaje + ciudadanas + ingles
    # matematicas 35% ciencias 35% lectura 15% sociales 10% ingles 5%
    ponderado = (matematicas*0.35)+(naturales*0.35)+(lenguaje*0.15)+(ciudadanas*0.1)+(ingles*0.05)
    
    data = {'ICFES': p_icfes,
            'PONDERADO': ponderado}
    features = pd.DataFrame(data, index=[0])
    
    st.sidebar.write('## Resultado total ICFES = ', p_icfes)
    st.sidebar.write('## Resultado de ponderado = ', ponderado)

    return features

def input_catastral():
    p_icfes = matematicas + naturales + lenguaje + ciudadanas + ingles
    # matematicas 35% ciencias 35% lectura 15% sociales 10% ingles 5%
    ponderado = (matematicas*0.35)+(naturales*0.30)+(lenguaje*0.15)+(ciudadanas*0.15)+(ingles*0.05)
    
    data = {'ICFES': p_icfes,
            'PONDERADO': ponderado}
    features = pd.DataFrame(data, index=[0])
    
    st.sidebar.write('## Resultado total ICFES = ', p_icfes)
    st.sidebar.write('## Resultado de ponderado = ', ponderado)

    return features



if carrera == 'Sistemas':
    df = input_ingenieria_general()
    # Reads in saved classification model
    load_clf = pickle.load(open('model-building/ing_sistemas_model.pkl', 'rb'))
    with col2:
        st.metric(label= 'tasa de acierto inteligencia aritficial', value='93.617 %', delta='-7.624 % de error de predicción')

elif carrera == 'Electrónica':
    df = input_ingenieria_general()
    load_clf = pickle.load(open('model-building/ing_electronica_model.pkl', 'rb'))

elif carrera == 'Eléctrica':
    df = input_ingenieria_general()
    load_clf = pickle.load(open('model-building/ing_electrica_model.pkl', 'rb'))

elif carrera == 'Industrial':
    df = input_ingenieria_general()
    load_clf = pickle.load(open('model-building/ing_industrial_model.pkl', 'rb'))

elif carrera == 'Catastral':
    df = input_catastral()
    st.write('hola')
    load_clf = pickle.load(open('model-building/ing_catastral_model.pkl', 'rb'))



if df['ICFES'][0] > 280:
    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    clasificacion = np.array(['ADMITIDO','NO ADMITIDO','OPCIONADO'])

    with col1:
        st.subheader('Resultado de predicción')
        estado_prediccion = str(clasificacion[prediction][0])
        st.title(estado_prediccion)

        if estado_prediccion == 'ADMITIDO':
            st.image(image_happy, caption='¡CelebrALO CuRRAMbA!')
        elif estado_prediccion == 'OPCIONADO':
            st.image(image_confused, caption='La fé es lo último que se pierde')
        else:
            st.image(image_sad, caption='¡No te desanimes! Vuélvelo a intentar')


    with col2:
        st.subheader('Probabilidad')
        st.write('Admitido:', round(prediction_proba[0][0]*100, 3), ' %')
        st.write('Opcionado:', round(prediction_proba[0][2]*100, 3), ' %')
        st.write('No admitido:', round(prediction_proba[0][1]*100, 3), ' %')
else:
    st.subheader('Puntaje menor al mínimo requerido (280 puntos) por la facultad de ingeniería')
    st.image(image_sad, caption='¡No te desanimes! Dios le da las peores batallas a sus mejores guerreros')