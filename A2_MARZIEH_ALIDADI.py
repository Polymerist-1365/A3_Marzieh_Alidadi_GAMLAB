# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:00:44 2024

@author: shamsabadi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl 

def Stress_Strain(data,application):
    '''
    this function converts F and dD to Stress and Strain by thickness(1.55mm), width(3.2mm) and parallel length(35mm).

    Parameters
    ----------
    data : DataFrame
        this DataFrame contains F(N) and dD(mm) received from the tensil test machine.
    application : str
        application determines the expected output of Stress_Strain function.

    Returns
    -------
    int, float or plot
        return may be elongation at break, strength or a plot.

    '''
    
    stress=np.array([data['F']/(1.55*3.2)])
    strain=np.array([(data['dD']/35)*100])
    if application.upper()=='ELONGATION AT BREAK':
        elongation_at_break=np.max(strain)
        print(elongation_at_break,'%')
        return elongation_at_break
    elif application.upper()=='STRENGTH':
        strength=np.max(stress)
        print(strength,'N/mm2')
        return strength
    elif application.upper()=='PLOT':
        myfont_title={'family':'sans-serif',
                      'color':'black',
                      'size':20}
        myfont_lables={'family':'Tahoma',
                       'color':'green',
                       'size':16}
        plt.plot(strain,stress,ls='--',c='g',linewidth=10)
        plt.title('Stress-Strain',fontdict=myfont_title)
        plt.xlabel('Strain(%)',fontdict=myfont_lables)
        plt.ylabel('Stress(N/mm2)',fontdict=myfont_lables)
        plt.show()
#****************function test******************        
titr=pd.DataFrame([[1,20],[2,34],[3,45],[4,67],[5,70],[4,89]],columns=['F','dD'])
Stress_Strain(titr,'plot')
Stress_Strain(titr,'elongation at break')
Stress_Strain(titr,'strength')


data_base=pd.read_excel('F-dD Data.xlsx')
tensile=pd.DataFrame(np.array(data_base),columns=['F','dD'])

#**********************clearing data***********************
tensile.info()
#out put: <class 'pandas.core.frame.DataFrame'>
#RangeIndex: 3890 entries, 0 to 3889
#Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
#---  ------  --------------  -----  
# 0   F       0 non-null      float64
# 1   dD      0 non-null      float64
#dtypes: float64(2)
#memory usage: 60.9 KB
# no empty cell and wrong format
count1=0
for i in tensile.index:
    if tensile.loc[i,'F']<0:
        count1=count1+1
        print(count1) #out put:0 -------> no wrong data in F column

count2=0
for j in tensile.index:
    if tensile.loc[j,'dD']<0:
        count2=count2+count1
        print(count2) #out put:0 -------> no wrong data in dD column
        
Stress_Strain(tensile,'elongation at break')
Stress_Strain(tensile,'strength')
Stress_Strain(tensile,'plot')
