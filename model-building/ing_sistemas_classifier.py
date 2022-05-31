import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import pickle


# read csv data from different years
df_2017_2 = pd.read_csv('./data/sistemas_2017-2.csv')
df_2018_1 = pd.read_csv('./data/sistemas_2018-1.csv')
df_2020_1 = pd.read_csv('./data/sistemas_2020-1.csv')


# combine datasets
frames = [df_2017_2, df_2018_1, df_2020_1]
data = pd.concat(frames)

# clean unnesary categorie
data=data[~data['RESULTADO'].isin(['INHABILITADO'])]


# map categories to numbers for trainning
lab_encoder = preprocessing.LabelEncoder()

# define the target
#y = data['RESULTADO']

y_num =  lab_encoder.fit_transform(data['RESULTADO'])

'''
mapeo = dict(zip(y, y_num))
print(mapeo)
print(y_num, lab_encoder.inverse_transform(y_num))
'''


#define features
features = ['ICFES', 'PONDERADO']
x = data[features].copy()

#split data
x_train, x_valid, y_train, y_valid = train_test_split(x, y_num, train_size = 0.8, test_size=0.2, random_state=0)

# define model classifier
model = RandomForestClassifier(n_estimators=500, class_weight={0:1, 1:2})
#model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_valid)


print(model.predict(pd.DataFrame({'ICFES': [343.00, 347.00, 355.00], 'PONDERADO': [69.90,68.90,72.70]})))

# get mean absolute error
error = mean_absolute_error(y_valid, y_pred)
accuaracy = accuracy_score(y_true=y_valid, y_pred=y_pred)
print('Error del modelo: ', error)
print('accuaracy: ', accuaracy)

# save model
pickle.dump(model, open('ing_sistemas_model.pkl', 'wb'))
