import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import pickle


# read csv data from different years
df_2017_2 = pd.read_csv('resultados_ing_sistemas_2017-2_anon.csv')
df_2018_1 = pd.read_csv('resultados_ing_sistemas_2018-1_anon.csv')
df_2020_1 = pd.read_csv('resultados_ing_sistemas_2020-1.csv')
df_2020_1.drop('N°', axis=1, inplace=True)

# combine datasets
frames = [df_2017_2, df_2018_1, df_2020_1]
data = pd.concat(frames)

# clean unnesary categorie
data=data[~data['ESTADO'].isin(['INHABILITADO'])]


# map categories to numbers for trainning
lab_encoder = preprocessing.LabelEncoder()

# define the target
y = data['ESTADO']

y_num =  lab_encoder.fit_transform(data['ESTADO'])


mapeo = dict(zip(y, y_num))
print(mapeo)
print(y_num, lab_encoder.inverse_transform(y_num))

#define features
features = ['PUNTAJE_ICFES', 'PUNTAJE_PONDERADO']
x = data[features].copy()

#split data
x_train, x_valid, y_train, y_valid = train_test_split(x, y_num, train_size = 0.8, test_size=0.2, random_state=0)

# define model classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_valid)


print(model.predict(pd.DataFrame({'PUNTAJE_ICFES': [343.00, 347.00, 355.00], 'PUNTAJE_PONDERADO': [69.90,68.90,72.70]})))

# get mean absolute error
error = mean_absolute_error(y_valid, y_pred)

print('Error del modelo: ', error)

# save model
pickle.dump(model, open('ing_sistemas_model.pkl', 'wb'))
