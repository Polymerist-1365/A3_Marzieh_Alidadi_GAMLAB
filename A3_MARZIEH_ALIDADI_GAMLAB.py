# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:43:18 2024

@author: shamsabadi
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#******************logesticregression************************************8
data=load_breast_cancer()
x=data.data
y=data.target
kf= KFold(n_splits=5,shuffle=True,random_state=42)
model=LogisticRegression()
my_params= {}
gs=GridSearchCV(model, my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_
gs.best_params_


#**********************************KNN**********************************8
data=load_breast_cancer()
x=data.data
y=data.target
kf= KFold(n_splits=5,shuffle=True,random_state=42)
model=KNeighborsClassifier() 
my_params= { 'n_neighbors':[1,5,10,15,20,25,30],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }
gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_ 
gs.best_params_


#**********************DT**************************************
data=load_breast_cancer()
x=data.data
y=data.target
kf= KFold(n_splits=5,shuffle=True,random_state=42)
model=DecisionTreeClassifier(random_state=42)
my_params={ 'max_depth':[1,5,8,10,15,20,25],
           'min_samples_split':[5,10,15],
           'min_samples_leaf':[1,2,3]}
gs=GridSearchCV(model, my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_
gs.best_params_


#********************************************RF**************************
data=load_breast_cancer()
x=data.data
y=data.target
kf= KFold(n_splits=5,shuffle=True,random_state=42)
model=RandomForestClassifier(random_state=42)
my_params={ 'n_estimators':[10,20,30,40,50],
           'max_features':[4,8,12]}
gs=GridSearchCV(model, my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_
gs.best_params_ 

#**********************************SVC*************************
data=load_breast_cancer()
x=data.data
y=data.target
kf= KFold(n_splits=5,shuffle=True,random_state=42)
model=SVC()
my_params={'kernel':['poly','rbf','linear'],
           'C':[0.001,0.01,1,10,50],
           'gamma':[0.001,0.01,0.1]}
gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy',return_train_score=True)
gs.fit(x,y)
gs.best_params_ 
gs.best_score_



#***************Final Report****************
'''
LogesticRegression
train_score: 0.9436619718309859
test_score: 0.965034965034965
'''


'''
KNN
gs.best_score_ 
Out[47]: np.float64(0.9419810588417947)
gs.best_params_
Out[48]: {'metric': 'manhattan', 'n_neighbors': 5}
'''

'''
DT
gs.best_score_
Out[79]: np.float64(0.9490141282409563)
gs.best_params_
Out[80]: {'max_depth': 5, 'min_samples_leaf': 3, 'min_samples_split': 10}
'''

'''
RF
gs.best_score_
Out[95]: np.float64(0.9630957925787922)
gs.best_params_ 
Out[96]: {'max_features': 4, 'n_estimators': 40}
'''

'''
SVC


'''
