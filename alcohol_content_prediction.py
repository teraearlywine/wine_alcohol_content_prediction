#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:22:17 2020

@author: teraearlywine
"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns

#wine machine learning project
from sklearn import datasets

wine = datasets.load_wine()
df = pd.DataFrame(data = np.c_[wine['data'], wine['target']],
                  columns= wine['feature_names']+['target'])


##Correlation to alcohol between each attribute of wine.

correlation = df.corr()['alcohol'].drop('alcohol')
print(correlation)


heat_map_of_correlation = sns.heatmap(df.corr())
plt.show()


### define a function to output features above a threshold value.

def get_features(correlation_threshold):
    abs_corrs = correlation.abs()
    high_correlations = abs_corrs
    [abs_corrs > correlation_threshold]
    new = pd.DataFrame(high_correlations).index.values.tolist()
    return new

#vector x containing input features and y containing alcohol variable
    
features = get_features(0.05)
print(features)

x = df[features]
y = df['alcohol']

#test_train setup
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)

#fit linear regression to training data
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#coefficients of 10 features w. highest alcohol content
regressor.coef_

train_pred = regressor.predict(x_train)
print(train_pred)

test_pred = regressor.predict(x_test)

print(test_pred)


###root mean squared error calculate
train_rmse = metrics.mean_squared_error(train_pred, y_train)** 0.5
print(train_rmse)

test_rmse = metrics.mean_squared_error(test_pred, y_test)**0.5
print(test_rmse)

#round off predicted values for test set
predicted_data = np.round_(test_pred)
print(predicted_data)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

# displaying coefficients of each feature

coeffecients = pd.DataFrame(regressor.coef_,features) 
coeffecients.columns = ['Coeffecient'] 
print(coeffecients)

