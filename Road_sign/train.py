# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn import svm
from xgboost import plot_importance
from matplotlib import pyplot

#Preprocessing
train = pd.read_csv("/home/abhishek/hackerearth/Road_sign/data/train.csv")
test = pd.read_csv("/home/abhishek/hackerearth/Road_sign/data/test.csv")

mapping = {'Front':0, 'Right':1, 'Left':2, 'Rear':3}
train = train.replace({'DetectedCamera':mapping})
test = test.replace({'DetectedCamera':mapping})

train.rename(columns = {'SignFacing (Target)': 'Target'}, inplace=True)

mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
train = train.replace({'Target':mapping})

y_train = np.array(train['Target'])
test_id = test['Id']

train.drop(['Target','Id','SignWidth'], inplace=True, axis=1)
test.drop(['Id','SignWidth'],inplace=True,axis=1)

train = np.array(train)
test = np.array(test)

#######
#Model Training Part
model = XGBClassifier(learning_rate=0.2)
model.fit(train, y_train)

#print(model.feature_importances_)
#plot feature importance
plot_importance(model)
pyplot.show()
# make predictions for test data

pred = model.predict_proba(test)
columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("xg_final2.csv", index=False)
