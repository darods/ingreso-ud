import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'resultados_ing_sistemas_anon.csv')
#print(data)
print('\n\n')
iris = datasets.load_iris()
print(data.head())
#print(type(iris))

data['Periodo_encoded'] = LabelEncoder().fit_transform(data['Periodo'])
data['Estado_encoded'] = LabelEncoder().fit_transform(data['ESTADO'])
print(data[['ESTADO', 'Estado_encoded']])
feature_cols=['Periodo_encoded', 'PUNTAJE_ICFES', 'PUNTAJE_PONDERADO']
X = data.loc[:, feature_cols]
print(X.shape)

Y = data.Estado_encoded
#print(y.shape)

clf = RandomForestClassifier()
clf.fit(X, Y)

print(data.Estado_encoded)