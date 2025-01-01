# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 20:09:33 2024

@author: shamsabadi
"""
import numpy as np
import matplotlib.pyplot as plt
#*******************************************numpy************************
a=np.array([[2,7,80,5],[3,13,5,9]])
b=np.array([[0,7,8,90],[13,9,45,76]])
c1=np.add(a,b)
print(c1)
c2=np.subtract(a,b)
print(c2)
c3=np.absolute(c2)
print(c3)
new1=np.concatenate((a,b),axis=0)
print(new1)
d1=np.mean(new1,axis=0)
print(d1)
d2=np.max(new1,axis=1)
print(d2)
new2=np.concatenate((a,b),axis=1)
print(new2)
d3=np.min(new2,axis=1)
print(d3)
d4=np.mean(new2[:,5])
print(d4)


#**********************************matplotlib*****************************
x=np.array([-5,-4,-2.5,-1,0,1,3,5,9,15,19,23,26,30,35])
y=4*(x**2)+3
plt.plot(x,y,marker='H')
plt.show()
plt.plot(x,y,marker='H',mec='g',mfc='y')
plt.show()
plt.plot(x,y,marker='H',mec='g',mfc='y',c='r')
plt.show()
plt.plot(x,y,marker='H',ms=15,mec='g',mfc='y',c='r')
plt.show()
plt.plot(x,y,marker='H',ms=5,mec='g',mfc='y',c='r',ls='-.')
