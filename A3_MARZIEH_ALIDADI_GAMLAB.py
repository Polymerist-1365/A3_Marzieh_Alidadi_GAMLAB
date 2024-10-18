
'''
APM:

Salam
daryaft shod besiar awli
moafagh bashid






'''

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:43:18 2024

@author: shamsabadi

"""

#-----------Import Libs----------------------

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


#-----------Import Data----------------------
data=load_breast_cancer()

#-----------Step1 : X and Y ----------------------
x=data.data
y=data.target
data.feature_names
data.target_names
#----------step2 : K fold split (cross validation)----------
#******************logesticregression************************************8

kf= KFold(n_splits=5,shuffle=True,random_state=42)
model=LogisticRegression()
my_params= {}
gs=GridSearchCV(model, my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_
gs.best_params_


#**********************************KNN**********************************8
model=KNeighborsClassifier() 
my_params= { 'n_neighbors':[1,5,10,15,20,25,30],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }
gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_ 
gs.best_params_


#**********************DT**************************************
model=DecisionTreeClassifier(random_state=42)
my_params={ 'max_depth':[1,5,8,10,15,20,25],
           'min_samples_split':[5,10,15],
           'min_samples_leaf':[1,2,3]}
gs=GridSearchCV(model, my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_
gs.best_params_


#********************************************RF**************************
model=RandomForestClassifier(random_state=42)
my_params={ 'n_estimators':[10,20,30,40,50],
           'max_features':[4,8,12]}
gs=GridSearchCV(model, my_params,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_score_
gs.best_params_ 

#**********************************SVC*************************
model=SVC()
my_params={'kernel':['poly','rbf','linear'],
           'C':[0.01,10,50]}
#My system is not powerful enough so when I add 'gamma' as a hyperparameter in my_params, the running of SVC model takes more than some days. So I would prefer to remove 'gamma' from my_params.
gs=GridSearchCV(model,my_params,cv=kf,scoring='accuracy',return_train_score=True)
gs.fit(x,y)
gs.best_score_
gs.best_params_ 



#***************Final Report****************

'''
FINAL REPORT :
    
Goal: The goal of this project is to be able to identify the type of breast cancer, whether benign or malignant, without the need for invasive sampling.

The inputs: The input data consists of the following columns represented as an array.These data are the physical properties of the cancerous gland of 500 patients in the form

array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'], dtype='<U23')

The output: The data provided indicates the type of breast cancer, whether it is benign or malignant.

array(['malignant', 'benign'], dtype='<U9')

Models: The following models were individually fitted to the input data:
    1. Logestic Regression (LR)
    2. K Nearest Neighbor (KNN)
    3. Desision Tree (DT)
    4. Random Forest (RF)
    5. Support Vector Classifier (SVC)
    
Best Score and Best Hiperparemeters: The best score and parameters of the above models after fitting to the input data were as follows:
    
LogesticRegression
gs.best_score_
Out[19]: np.float64(0.9419655333022823)
gs.best_params_
Out[20]: {}

KNN
gs.best_score_ 
Out[22]: np.float64(0.9419810588417947)
gs.best_params_
Out[23]: {'metric': 'manhattan', 'n_neighbors': 5}

DT
gs.best_score_
Out[25]: np.float64(0.9490141282409563)
gs.best_params_
Out[26]: {'max_depth': 5, 'min_samples_leaf': 3, 'min_samples_split': 10}

RF
gs.best_score_
Out[28]: np.float64(0.9630957925787922)
gs.best_params_ 
Out[29]: {'max_features': 4, 'n_estimators': 40}

SVC
gs.best_score_
Out[16]: np.float64(0.9612793044558299)

gs.best_params_ 
Out[17]: {'C': 50, 'kernel': 'linear'}

Model performance comparison:A comparison between the best scores of the models shows that the RF model with the best score of 0.9630957925787922 has the best performance in predicting the type of breast cancer after inputting the physical properties of the cancerous gland.

'''
