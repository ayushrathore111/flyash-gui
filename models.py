
import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.neural_network   import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# Read the CSV file
data = pd.read_excel('DEF FULL RCA.xlsx')
data = data.dropna()
from sklearn.neighbors import KNeighborsRegressor

# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None)
lr_model = LinearRegression()
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
knn_model = KNeighborsRegressor(n_neighbors=5)
etr_model = ExtraTreesRegressor(random_state=42)
dtr_model = DecisionTreeRegressor(random_state=42)
adaboost_model = AdaBoostRegressor(random_state=42)
bagging_model = BaggingRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

knn_model.fit(X_train,y_train)
etr_model.fit(X_train,y_train)
dtr_model.fit(X_train,y_train)
bagging_model.fit(X_train,y_train)
adaboost_model.fit(X_train,y_train)
xgb_model.fit(X_train,y_train)
lr_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)
gbr_model.fit(X_train,y_train)
svr_model.fit(X_train,y_train)
import joblib

joblib.dump(knn_model, './static/knn.joblib')
joblib.dump(etr_model, './static/etr.joblib')
joblib.dump(bagging_model, './static/br.joblib')
joblib.dump(adaboost_model, './static/ar.joblib')
joblib.dump(xgb_model, './static/xg.joblib')
joblib.dump(rf_model, './static/rf.joblib')
joblib.dump(dtr_model, './static/dtr.joblib')
joblib.dump(lr_model, './static/lr.joblib')
joblib.dump(gbr_model, './static/gbr.joblib')
joblib.dump(svr_model, './static/svr.joblib')