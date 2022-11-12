# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:55:52 2022

@author: kiran
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import pickle 
df = pd.read_csv(r'C:/Users/kiran/Downloads/docker_test/Cars.csv')
df.head()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


labelencoder.fit(df['Enginetype'])
pickle.dump(labelencoder,open('labelencoder.pkl','wb'))
# Creating labels for each columns 
df['Enginetype'] = labelencoder.transform(df['Enginetype'])
# label encode y ####


# Concatenate x and y 

# splitting the data into the columns which need to be trained(X) and the target column👍
X =df.drop(labels=["MPG"],axis=1)
y = df.MPG
  
# splitting data into training and testing data with 30 % of data as testing data respectively

X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# import library to perform multilinear regression

multilinear = LinearRegression()

multilinear.fit(X_train, y_train)
pickle.dump(multilinear,open('mlr.pkl','wb'))
model=pickle.load(open('mlr.pkl','rb'))
X_train

# Predicting upon X_test
y_pred = model.predict(X_test)
y_pred

# checking the Accurarcy by using r2_score
accuracy = r2_score(y_test, y_pred)
accuracy

X.shape


og=pickle.load(open('labelencoder.pkl','rb'))

engine = og.transform(['hybrid'])
print(engine)
a=engine.reshape(-1,1)

yp = model.predict([[1,60,60,120,33]])
print(model.predict([[og.transform(['hybrid']), 130,   86,  127.909442 , 28.070597]]))

yp
