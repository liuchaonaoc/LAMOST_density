#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 11:55:45 2016

@author: chaoliu
"""
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import time
import LM_density as lm
# Load LAMOST data file
#from LM_density import readDR3
#from LM_density import getHaloRGB


if __name__ == '__main__':
    # Read the selection function data file for all DR3 plates
    S0 = np.genfromtxt(
        'Selection_plates.csv',           # file name
        skip_header=0,          # lines to skip at the top
        skip_footer=0,          # lines to skip at the bottom
        delimiter=',',          # column delimiter
        dtype='float32',        # data type
        filling_values=0)       # fill missing values with 0
    plateid = S0[:,0]
    dK = 0.25
    dJK = 0.1
    Kgrid = np.arange(0,15+dK,dK)
    JKgrid = np.arange(-0.5,4+dJK,dJK)
    
    # read DR3 data
    D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3 = lm.readDR3()
    
    # halo RGB sample
    ind_hRGB = (D>0) & (dr3.M_K50<-3.5) & (dr3.feh<-1) & (dr3.RGB>0)
    D_hRGB, Dlow_hRGB, Dup_hRGB, X_hRGB, Y_hRGB, Z_hRGB, R_hRGB, \
        r_hRGB, K_hRGB, JK_hRGB, plateserial_hRGB = lm.getPop(D,\
        Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial,ind_hRGB)
    
        # derive nu for haloRGB sample
    dD=0.01
    Dgrid = np.arange(0,200+dD,dD)
    nu_hRGB = lm.nulall(S0,K_hRGB,JK_hRGB,D_hRGB, Dlow_hRGB,Dup_hRGB,\
                 plateserial_hRGB, Kgrid, JKgrid, dK, dJK, Dgrid)
    
    lm.save_file(dr3,ind_hRGB,D_hRGB,Dlow_hRGB,Dup_hRGB,Z_hRGB,\
             R_hRGB,r_hRGB,np.log(nu_hRGB),'LMDR3_haloRGB.dat')
    
    lm.test_nu(R_hRGB, Z_hRGB, nu_hRGB)