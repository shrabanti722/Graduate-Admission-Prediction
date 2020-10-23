import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=X,label=y)

regressor = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
regressor.fit(X_train, y_train)

#regressor = xgb.XGBRegressor(objective="reg:linear", random_state=42)

plot_importance(regressor)
pyplot.show()

y_pred = regressor.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
'''xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()'''

regressor.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()



'''from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

from xgboost import XGBClassifier
regressor = XGBClassifier()
regressor.fit(X_train, y_train)'''

'''
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred)
print(np.sqrt(mse))

y_pred = regressor.predict(X_test)
predictions = [round(value) for value in y_pred]
y_test_1 = [round(value) for value in y_test]
accuracy = accuracy_score(y_test_1, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_1, predictions)

#y_pred = regressor.predict(X_test)

regressor.score(X_test, y_test)

from sklearn.metrics import accuracy_score
accuracy_score( y_test,y_pred)
'''
