# Compare between Standard scaler Regression VS Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets.samples_generator import make_regression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler

train_data = np.genfromtxt("./data/regression/trainingData.csv", delimiter=',')
test_data = np.genfromtxt("./data/regression/testData.csv", delimiter=',')

X_train = train_data[:,0:12]
y_train = train_data[:,12:13]

X_test = test_data[:,0:12]
y_test = test_data[:,12:13]

reg = KNeighborsRegressor()
reg = reg.fit(X_train, y_train)

results= reg.predict(X_test)
print (metrics.r2_score(y_test,results))




scaler = StandardScaler()
scaler.fit(X_train)

scaled_features_Train = scaler.transform(X_train)
scaled_features_Test = scaler.transform(X_test)


reg = KNeighborsRegressor()
reg = reg.fit(scaled_features_Train, y_train)

results= reg.predict(scaled_features_Test)
print (metrics.r2_score(y_test,results))
