import os
import sys
import pickle
from tqdm import tqdm 

import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    os.mkdir('models')
except:
    pass

data = pd.read_csv('Iris.csv')


def Encoding(x):
    if x == 'Iris-setosa':
        return 0
    elif x == 'Iris-versicolor':
        return 1
    else:
        return 2

data['labels'] = data['Species'].apply(lambda x : Encoding(x))
print(data.head())

trainDF = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
target = data['labels']

x_train,x_test,y_train,y_test = train_test_split(trainDF,target)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print('Accuracy Score : ', accuracy_score(y_test,y_pred))
pickle.dump(model,open('models/LogisticClassifier.pkl','wb'))
