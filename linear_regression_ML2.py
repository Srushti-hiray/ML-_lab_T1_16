# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:11:31 2024

@author: icon
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

df=pd.read_csv("C:/CSV files/50_Startups.csv")
df.columns

df.corr()
df["State"].unique()

labelencoder=preprocessing.LabelEncoder()
df["State"]=labelencoder.fit_transform(df["State"])

sns.pairplot(df, x_vars=[ 'Marketing Spend', 'State','R&D Spend','Administration' ], y_vars = 'Profit', size = 4, kind = 'scatter' )
plt.show()

X=df.drop(columns=["Profit"])
y=df["Profit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

print( X_train.shape )
print( X_test.shape )
print( y_train.shape )
print( y_test.shape )


regressor = LinearRegression()
a=regressor.fit(X_train, y_train)

r_sq = a.score(X, y)

print(regressor.score(X_train,y_train))
print(regressor.score(X_test,y_test))

print(f"coefficient of determination: {r_sq}")

print(f"intercept: {a.intercept_}")

print(f"slope: {a.coef_}")


y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_test
df_new = pd.DataFrame({"actual":y_test,"predicted:":y_pred_test})
df_new
y_pred_train = regressor.predict(X_train)   # predicted value of y_train
y_pred_train

print("mean absolute error")
print(metrics.mean_absolute_error(y_test,y_pred_test))

print("mean square error")
print(metrics.mean_squared_error(y_test,y_pred_test))












