# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:24:55 2024

@author: shamsabadi
"""


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#****************************************LogesticRegression**********************
iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
# 0.9642857142857143
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 
#1.0
#**************************KNN***************************************

#*************************n_neighbors=5********************************
iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#0.9642857142857143
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score)
#1.0

#*************************n_neighbors=30********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=KNeighborsClassifier(n_neighbors=30)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#0.9464285714285714
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score)
#1.0

#*************************n_neighbors=50********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=KNeighborsClassifier(n_neighbors=50)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#0.875
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score)
#0.9736842105263158
#****************************DecisionTree*************************

#*************************max_depth=2********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=DecisionTreeClassifier(max_depth=2,random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#0.9464285714285714
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 
#0.9736842105263158

#*************************max_depth=4********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=DecisionTreeClassifier(max_depth=4,random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#0.9732142857142857
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 
#1.0

#*************************max_depth=6********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=DecisionTreeClassifier(max_depth=6,random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#1.0
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 
#1.0
#************************************RandomForest***********************

#*************************n_estimators=30********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=RandomForestClassifier(random_state=42,n_estimators=30)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#1.0
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 
#1.0

#*************************n_estimators=50********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=RandomForestClassifier(random_state=42,n_estimators=50)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#1.0
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 
#1.0

#*************************n_estimators=80********************************

iris=load_iris()
x=iris.data
print(iris.feature_names)
y=iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=42)
model=RandomForestClassifier(random_state=42,n_estimators=80)
model.fit(x_train,y_train)

y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 
#1.0
y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 
#1.0

#*************************END*********************************************
