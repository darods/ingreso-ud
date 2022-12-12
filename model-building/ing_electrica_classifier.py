import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pickle


# read csv data from different years
df_2018_1 = pd.read_csv('../data/electrica_2018-1.csv')
df_2022_1 = pd.read_csv('../data/electrica_2022-1.csv')


# combine datasets
frames = [df_2018_1, df_2022_1]
data = pd.concat(frames)

# clean unnesary categorie
data=data[~data['RESULTADO'].isin(['INHABILITADO'])]


# map categories to numbers for trainning
lab_encoder = preprocessing.LabelEncoder()

y_num =  lab_encoder.fit_transform(data['RESULTADO'])

#define features
features = ['ICFES', 'PONDERADO']
x = data[features].copy()

#split data
x_train, x_valid, y_train, y_valid = train_test_split(x, y_num, train_size = 0.8, test_size=0.2, random_state=0)

# Random Forest Classifier
modelRF = RandomForestClassifier(n_estimators=500, class_weight={0:1, 1:2})
modelRF.fit(x_train, y_train)
y_predRF = modelRF.predict(x_valid)
errorRF = mean_absolute_error(y_valid, y_predRF)
accuaracyRF = accuracy_score(y_true=y_valid, y_pred=y_predRF)

# K Neighbors Classifier
modelKN = KNeighborsClassifier()
modelKN.fit(x_train, y_train)
y_predKN = modelKN.predict(x_valid)
errorKN = mean_absolute_error(y_valid, y_predKN)
accuaracyKN = accuracy_score(y_true=y_valid, y_pred=y_predKN)

# GaussianNB
modelGNB = GaussianNB()
modelGNB.fit(x_train, y_train)
y_predGNB = modelGNB.predict(x_valid)
errorGNB = mean_absolute_error(y_valid, y_predGNB)
accuaracyGNB = accuracy_score(y_true=y_valid, y_pred=y_predGNB)

# Decision Tree Classifier
modelDT = DecisionTreeClassifier()
modelDT.fit(x_train, y_train)
y_predDT = modelDT.predict(x_valid)
errorDT = mean_absolute_error(y_valid, y_predDT)
accuaracyDT = accuracy_score(y_true=y_valid, y_pred=y_predDT)

# Support Vector Classifier
modelSVC = SVC()
modelSVC.fit(x_train, y_train)
y_predSVC = modelSVC.predict(x_valid)
errorSVC = mean_absolute_error(y_valid, y_predSVC)
accuaracySVC = accuracy_score(y_true=y_valid, y_pred=y_predSVC)

#print(model.predict(pd.DataFrame({'ICFES': [343.00, 347.00, 355.00], 'PONDERADO': [69.90,68.90,72.70]})))

# Get MSE
print('ingenieria electrica')
print('\nErrores MSE')
print('Random Forest Classifier: ', errorRF)
print('K Neighbors Classifier: ', errorKN)
print('GaussianNB: ', errorGNB)
print('Decision Tree Classifier: ', errorDT)
print('Support Vector Classifier: ', errorSVC)

print('\nAccuaracy')
print('Random Forest Classifier: ', accuaracyRF)
print('K Neighbors Classifier: ', accuaracyKN)
print('GaussianNB: ', accuaracyGNB)
print('Decision Tree Classifier: ', accuaracyDT)
print('Support Vector Classifier: ', accuaracySVC)

# save model
pickle.dump(modelSVC, open('ing_electrica_model.pkl', 'wb'))
