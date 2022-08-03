#!/usr/bin/env python                                                   
# -*- coding: utf-8 -*-                                                 
'''                                                                     
this script is intended to apply different acceleration module including numba and cupy defined in this 
directory/smooth_functionstore.py,you could select and compare to get a clue for future extension.
Cupy saved half time of the numba running,which has already improved a lot from my previous version.
In the new function of numba, we packed several vector into one matrix and try best to vectorise and numpy
functions and saved a lot of time.
''' 

#from pyexodus import exodus                                             
import numpy as np                                                      
#import h5py                                                             
import os    
#import numba
import datetime
import cupy as cp
#import lasif


    
def LBFGS_n_order(gradient,model):
    order,l2=np.shape(gradient)
    gama=np.zeros((order-1,l2))
    s=np.zeros((order-1,l2))
    r=np.zeros(order-1)
    for i in np.arange(0,order-1):
        gama[i]= gradient[i+1] - gradient[i] ###gama=gradient6-gradient5
        s[i]   = model[i+1]    -  model[i] ####s5= model6-model5
        r[i]   = 1/np.inner(gama[i],s[i])
    q2=gradient[order-1]
    alpha=np.zeros(order-1)
    for i in range(order-1):
        print(i)
        index=np.int(order-2-i)
        print(index)
        alpha[index]=r[index]*np.dot(s[index],q2)
        q2=q2-alpha[index]*gama[index]        
    z=q2*np.dot(s[order-2],gama[order-2])/np.dot(gama[order-2],gama[order-2])
    for i in range(order-1):  
        belta=r[i]*np.inner(gama[i],z)
        z=z+s[i]*(alpha[i]-belta)
    return z

