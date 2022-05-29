import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.write("""
# Aplicación para la predicción de candidato a la **UD** (en desarrollo!)
""")

st.header('Parametros del usuario')

def user_input_features():
    periodo = st.sidebar.slider('Periodo', 0, 3, 1)
    p_icfes = st.sidebar.slider('PUNTAJE_ICFES', 280.00, 500.00, 370.00)
    ponderado = st.sidebar.slider('PUNTAJE_PONDERADO', 0.00, 100.00, 70.00)
    #sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    #sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    #petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    #petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'Periodo':periodo,
            'PUNTAJE_ICFES': p_icfes,
            'PUNTAJE_PONDERADO': ponderado}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv(r'resultados_ing_sistemas_anon.csv')
data['Periodo_encoded'] = LabelEncoder().fit_transform(data['Periodo'])
data['Estado_encoded'] = LabelEncoder().fit_transform(data['ESTADO'])
feature_cols=['Periodo_encoded', 'PUNTAJE_ICFES', 'PUNTAJE_PONDERADO']
X = data.loc[:, feature_cols]
Y = data.Estado_encoded

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
#st.write(data.Estado_encoded)

st.subheader('Prediction')
#st.write(data.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)