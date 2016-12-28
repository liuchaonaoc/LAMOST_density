#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 11:01:23 2016

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
    ind_dRGB = (dr3.WHF_sample2==84) & (D>0)
    
    D_dRGB, Dlow_dRGB, Dup_dRGB, X_dRGB, Y_dRGB, Z_dRGB, R_dRGB, \
        r_dRGB, K_dRGB, JK_dRGB, plateserial_dRGB = lm.getPop(D,\
        Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial,ind_dRGB)
    
        # derive nu for haloRGB sample
    dD=0.01
    Dgrid = np.arange(0,200+dD,dD)
    nu_dRGB = lm.nulall(S0,K_dRGB,JK_dRGB,D_dRGB, Dlow_dRGB,Dup_dRGB,\
                 plateserial_dRGB, Kgrid, JKgrid, dK, dJK, Dgrid)
    
    lm.save_file(dr3,ind_dRGB,D_dRGB,Dlow_dRGB,Dup_dRGB,Z_dRGB,\
             R_dRGB,r_dRGB,np.log(nu_dRGB),'LMDR3_diskRGB.dat')
    
    lm.test_nu(R_dRGB, Z_dRGB, nu_dRGB)