# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:06:30 2024

@author: shamsabadi
"""

import math
#***********PART1**********CONSTANT NUMBERS*******************************
N=6.02214*math.pow(10,23)
#Avogadro Number, 1/mole
R=8.3145
#Universal Gas Constant, j/(mol.K)
Kb=1.380649*math.pow(10,-23)
#Boltzmann Constant, j/K
Ksb=5.67*math.pow(10,-8)
#Stephan-Boltzmann Constant, W/(k^4.m^2)
h=6.6236*math.pow(10,-36)
#Planck Constant, j.s


#***********PART2***********FUNCTIONS********************
#William, Landel, Ferrry (WLF)
def WLF(T,Tg,/):
    '''
    The WLF equation is a procedure for shifting data for amorphous polymers obtained at elevated temperatures to a reference temperature. 

    Parameters
    ----------
    T : int or float
        Temperature, K or degree celsius, Tg<T<Tg+100..
    Tg : int or float
        Glass Transition Temperature, K or degree celsius.
   
    Returns
    -------
    aT : int or float
    shift factor.

    '''
    b=T-Tg 
    c=-17.44*b
    d=51.6+b
    e=c/d
    aT=math.pow(10,e)
    return aT
        

#Cohen Equation
def Cohen(m,r,/):
    '''
  Cohen equation is used to predict the yield stress of a polymeric blend  containing a rubbery dispersion phase. 

    Parameters
    ----------
    m : int or float
        The yield stress of polymeric matrix, N/m^2 or Pa or ...
    r : float
        Volume fraction of rubbery phase, 0<r<1.

    Returns
    -------
    b : int or float
    The yield stress of polymeric blend, N/m^2 or Pa or ...

    '''
    a=(1-1.21*math.pow(r,(2/3)))
    b=m*a
    return b
    
        
#Critical diameter of rubbery particles
def Critical_Diameter(d,r,/):
    '''
    This equation predicts the critical diameter of rubber particles toughening a polymeric matrix.
    Parameters
    ----------
    d : int or float
        critical distance between rubbery particles, angstrom or mm or nm or .....
    r : float
        Volume fraction of rubbery phase, 0<r<1.

    Returns
    -------
    dc : int or float
    the critical diameter of rubber particles

    '''
    a=6*math.pow(r,(1/3))
    b=a-1
    c=3.14/b
    dc=d/c
    return dc


#**************PART3**********CONVERTOR************************************8
def Frequency_Conv1(a,/):
    '''
    A converter machine to convert frequency in Hertz(Hz) to frequency in rpm.
    Parameters
    ----------
    a : int or float
        frequency, Hertz(Hz).

    Returns
    b : int or float 
    frequency, revolution per minute (rpm)
    '''
    b=a*60
    return b


#************************************************************
def Frequency_Conv2(b,/):
    '''
   A converter machine to convert frequency in rpm to frequency in Herta(Hz).
    Parameters
    ----------
    b : int or float
        frequency, revolution per minute (rpm).

    Returns
    a, frequency, Hertz(Hz)

    '''
    a=b/60
    return a
    
   

