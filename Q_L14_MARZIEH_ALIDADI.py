# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:48:19 2024

@author: shamsabadi
"""
'''
x is the weight ratio of liquid phase (water+alcohol) to solid phase (cellulose) in the process of carboxymethyl cellulose (CMC) synthesis
y is the dynamic viscosity of CMC aqueous solution (2%) in environmental conditions
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
model=LinearRegression()
a=np.array([ [3.17,118.33],[5,220.09],[6,266.67],[7,98.14],[8.83,205.67] ])
cmc=pd.DataFrame(a,columns=['ratio','viscosity'])
x=np.array(cmc['ratio']).reshape(-1,1)
y=np.array(cmc['viscosity'])
model.fit(x,y)
A=model.coef_
B=model.intercept_
print(A)
print(B)
#y=6.94991619*x+140.08050283608432
#viscosity=6.94991619*ratio+140.08050283608432
new_x=np.linspace(0,100,1000).reshape(-1,1)
y_pred=A*new_x+B
plt.scatter(x,y,s=20,label='Experimental')
plt.scatter(new_x,y_pred,s=5,label='Predicted')
plt.title('Reaction Medium Effect')
plt.xlabel('Ratio (liquid phase/cellulose)')
plt.ylabel('Viscosity (cp)')
plt.grid()
plt.legend()
plt.show()

#*****************train-test***************************************
model=LinearRegression()
train=np.array([[3.17,118.33],[5,220.09],[6,266.67],[7,98.14]])
test=np.array([[8.83,205.67]])
x_train=train[:,0]
x_train=np.array(x_train).reshape(-1,1)
y_train=train[:,1]
y_train=np.array(y_train)
x_test=test[:,0]
y_test=test[:,1]
model.fit(x_train,y_train)
y_pred=model.predict(x_test.reshape(-1,1))
MAE=mean_absolute_error(y_test,y_pred)
print(MAE)
MAPE=mean_absolute_percentage_error(y_test,y_pred)
print(MAPE)


#****************END*****************************