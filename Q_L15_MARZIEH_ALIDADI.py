# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:57:55 2024

@author: shamsabadi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



#*****************************LinearRegression******************
'''
x is the molar ratio of monochloro acetic acid (MCA) to anhydroUS-glocuse in the second step of carboxymethyl cellulose (CMC) proccessing.
y in the degree of substitution (DS) of carboxyl group on the glocouse ring in cellulose chain.
'''
data=pd.DataFrame([[0.9,0.78],[0.95,0.79],[1.0,0.76],[1.1,0.63],
                [1.2,0.85],[1.3,1.05],[1.5,0.98],[1.6,0.82],[1.7,0.99],[1.55,0.88],[1.05,0.58],[1.65,0.99]],columns=['concentration','DS'])

x=np.array(data['concentration']).reshape(-1,1)
y=np.array(data['DS'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,shuffle=True,random_state=50)

model=LR()
model.fit(x_train,y_train)
A=model.coef_
B=model.intercept_
print(A)
print(B)
#y=0.28738197*x+0.4923690987124466
x_new1=np.arange(40,100)
x_new=(x_new1/100)*2
print(x_new)
y_pred=A*x_new+B
plt.scatter(x_train,y_train,label='training points')
plt.scatter(x_test,y_test,label='testing points')
plt.plot(x_new,y_pred,label='predicted line')
plt.title('MCA-DS')
plt.xlabel('MCA/anhydrous glucose')
plt.ylabel('DS')
plt.grid()
plt.legend()
plt.show()

y_train_pred=model.predict(x_train.reshape(-1,1))

training_score1=MAE(y_train,y_train_pred)
print('MAE train score model: ', training_score1)
training_score2=MAPE(y_train,y_train_pred)
print('MAPE train score model: ', training_score2)

y_test_pred=model.predict(x_test.reshape(-1,1))

test_score1=MAE(y_test,y_test_pred)
print('MAE test score:' , test_score1 )
test_score2=MAPE(y_test,y_test_pred)
print('MAPE test score:' , test_score2 )


#*******************Logesticregression**********************
'''
x1 is the weigth of fruit.------->weigth
x2 is the diameter of fruit.------>diameter

output:
    1------->watermelon
    0------->apple
'''
data=pd.DataFrame([[4500,21,1],[250,10,0],[3200,19,1],[2700,17.5,1],[2950,18,1],[170,6.5,0],[156,6,0],[160,6.4,0],[220,9.2,0],[3800,19.9,1],[4100,20,1],[4400,20.5,1],[280,11.3,0],[305,11.8,0],[290,11.4,0],[3650,18.7,1]],
                  columns=['weigth','diameter','type of fruit'])

x=np.array(data[['weigth','diameter']])
y=np.array(data['type of fruit'])

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, shuffle=True, random_state=65)
model=LogisticRegression()
model.fit(x_train,y_train)

new_fruit=np.array([[3645,18.23]])
model.predict(new_fruit)
#array([1])

plt.scatter(x_train[:,0],x_train[:,1],c=y_train , cmap='viridis')
plt.scatter(x_test[:,0],x_test[:,1],c=y_test , cmap='viridis')
plt.title('Fruit Typer')
plt.xlabel('weigth(gr)')
plt.ylabel('diameter(cm)')
plt.show()

y_train_pred=model.predict(x_train)
train_score1=confusion_matrix(y_train,y_train_pred)
train_score2=accuracy_score(y_train,y_train_pred)
print(train_score1)
#[[8 0]
#[0 4]]
print(train_score2)
#1.0

y_test_pred=model.predict(x_test)
test_score1=confusion_matrix(y_test,y_test_pred)
test_score2=accuracy_score(y_test,y_test_pred)
print(test_score1)
#[[4]]
print(test_score2)
#1.0

#*************************************END********************************