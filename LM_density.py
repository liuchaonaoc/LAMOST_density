#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 21:36:29 2016

@author: chaoliu
"""

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import time

## calculate setllar density at the positions in which the stars are located
#def extract_nu():
#    nu_i=np.zeros((len(D),1))
#    for i in range(len(D)):
#        ii = plateserial(i)
#        if isempty(nu_S[ii]):
#                indnu=~isnan(nu_S[ii]) & ~isinf(nu_S[ii]) & nu_S[ii]>0;
#                if np.sum(indnu)>2:
#                    nu_i[i]=interp1(Dgrid[indnu],nu_S[ii][indnu],D(i));
#    return nu_i

# gal2cart
def gal2cart(l,b,D,Rsun,zsun):
    cl = np.cos(l*np.pi/180.0)
    sl = np.sin(l*np.pi/180.0)
    cb = np.cos(b*np.pi/180.0)
    sb = np.sin(b*np.pi/180.0)
    x = D*cl*cb-Rsun #x=0 at GC; x=-Rsun at the Sun
    z = zsun+D*sb
    y = D*sl*cb #positive point to rotation direction
    return x,y,z


def readDR3():
    hdulist = fits.open('/Users/chaoliu/mw/lamost_regular/data/DR3/DR3_b1_indspv1_photo_dist_short2.fits')
    dr3 = hdulist[1].data
    # XYZ
    R0=8000 #pc the distance from the Sun to the GC
    Z0=27 # height of the Sun above mid-plane
    D = dr3.distK50_RJCE/1000.0 #kpc
    Dlow = D-dr3.distK85_RJCE/1000.0 #kpc
    Dup = dr3.distK15_RJCE/1000.0-D #kpc
    
    X,Y,Z = gal2cart(dr3.glon,dr3.glat,dr3.distK50_RJCE,R0,Z0) #R0=8000
    R = np.sqrt(X**2+Y**2)
    r_gc = np.sqrt(R**2+Z**2)
    # in kpc
    X = X/1000.0
    Y = Y/1000.0
    Z = Z/1000.0
    R = R/1000.0
    r_gc = r_gc/1000.0
    K = dr3.Kmag_2mass
    JK = dr3.Jmag_2mass-dr3.Kmag_2mass
    plateserial = dr3.plateserial
    #MK = dr3.M_K50
    #feh = dr3.feh
    
    return D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3

def getHaloRGB(D,Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, ind_haloRGB): 
    # halo RGB for XU et al.
    
    D_hRGB = D[ind_haloRGB]
    Dlow_hRGB = Dlow[ind_haloRGB]
    Dup_hRGB = Dup[ind_haloRGB]
    X_hRGB = X[ind_haloRGB]
    Y_hRGB = Y[ind_haloRGB]
    Z_hRGB = Z[ind_haloRGB]
    R_hRGB = R[ind_haloRGB]
    r_hRGB = r_gc[ind_haloRGB]
    K_hRGB = K[ind_haloRGB]
    JK_hRGB = JK[ind_haloRGB]
    #glon_hRGB = dr3.glon[ind_haloRGB]
    #glat_hRGB = dr3.glat[ind_haloRGB]
    plateserial_hRGB = plateserial[ind_haloRGB]

    return D_hRGB, Dlow_hRGB, Dup_hRGB, X_hRGB, Y_hRGB, \
        Z_hRGB, R_hRGB, r_hRGB, K_hRGB, JK_hRGB, plateserial_hRGB
        
### functions to calculate the los density profile and all density profiles
# calculate stellar density along a line of sight
def nu_los(m,c,D,Dlow,Dup,Smap,Dgrid,mgrid,cgrid,dm,dc):
    N = len(m)
    im = np.array([np.int(i) for i in np.round((m-mgrid[0])/dm)])
    ic = np.array([np.int(i) for i in np.round((c-cgrid[0])/dc)])
    
    #print im,(im<0)
    im[im<0] = 0
    im[im>=len(mgrid)] = len(mgrid)-1
    ic[ic<0] = 0
    ic[ic>=len(cgrid)] = len(cgrid)-1
    #print ic,im
    Nm = len(mgrid)
    #print np.shape(Smap)
    S = Smap[im+ic*Nm]
    #print c,m
    nu = np.zeros(np.shape(Dgrid))
    for i in range(N):
        pp1 = np.exp(-(D[i]-Dgrid)**2/(2*Dlow[i]**2))
        pp2 = np.exp(-(D[i]-Dgrid)**2/(2*Dup[i]**2))
        pp = pp1
        pp[Dgrid>D[i]] = pp2[Dgrid>D[i]];
        
        nu0 = (pp/np.sum(pp)/(Dgrid**2))*S[i]*dm*dc
        #print np.shape(pp),np.shape(S),np.shape(nu+nu0.reshape(np.shape(nu)))
        nu = nu+nu0#nu0.reshape(np.shape(nu));
    return nu
                
# calculate stellar density for all lines of sight
def nulall(S0,K,JK,D, Dlow,Dup,plateserial, Kgrid, JKgrid, dK, dJK, Dgrid):
    start = time.clock()
    nu_i = np.zeros(np.shape(D))
    Nplate = np.shape(S0)
    for i in range(Nplate[0]):
        #indP = plates.pid==i
        pid = S0[i,0]
        indStar = (plateserial==pid) & (D>0) & (D<Dgrid[-1])
        #print np.sum(indStar)
        if np.sum(indStar)>0:
            nu1 = nu_los(K[indStar],JK[indStar],\
            D[indStar],\
            Dlow[indStar],Dup[indStar],\
            S0[i,1:],Dgrid,Kgrid,JKgrid,dK,dJK)
            indnu = (~np.isinf(nu1)) & (nu1>0) & (~np.isnan(nu1))
            if np.sum(indnu)>2:
                #iD=np.array([np.int(ii) for ii in np.round((D[indStar]-Dgrid[0])/0.01) ])
                #nu_i[indStar]=nu1[iD]
                f = interp1d(Dgrid[indnu],nu1[indnu],bounds_error=False,fill_value=0)
                nu_i[indStar] = f(D[indStar])
            print i
    print 'Time=%(t).8f' % {'t':time.clock()-start}
    return nu_i
    

def test_nu(R,Z,nu):
    Rgrid = np.arange(0,100.,2.)
    Zgrid = np.arange(0,100.,2.)
    iR = np.array([np.int(i) for i in np.round((R-Rgrid[0])/2.)])
    iZ = np.array([np.int(i) for i in np.round((np.abs(Z)-Zgrid[0])/2.)])
    ind = (iR>=0) & (iR<len(Rgrid)) & (iZ>=0) & (iZ<len(Zgrid))
    Rzmap = np.zeros((len(Rgrid),len(Zgrid)))
    NRzmap = np.zeros((len(Rgrid),len(Zgrid)))
    for i in range(len(R)):
        if ind[i]>0 and ~np.isnan(nu[i]) and ~np.isinf(nu[i]) and nu[i]>0:
            Rzmap[iR[i],iZ[i]] = Rzmap[iR[i],iZ[i]]+nu[i]
            NRzmap[iR[i],iZ[i]] = NRzmap[iR[i],iZ[i]]+1
    Rzmap = Rzmap / NRzmap    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.log(Rzmap.T),vmin=-18,vmax=-9,interpolation='nearest',\
              extent=[0,100,100,0])
    ax.set_ylim([0,80])
    ax.set_xlim([0,80])
    
    Rmesh,Zmesh = np.meshgrid(Rgrid,Zgrid)
    Rmesh = Rmesh.reshape((len(Rgrid)*len(Zgrid),))
    Zmesh = Zmesh.reshape((len(Rgrid)*len(Zgrid),))
    numesh = Rzmap.reshape((len(Rgrid)*len(Zgrid),))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot((np.sqrt(Rmesh**2+Zmesh**2)),np.log(numesh),'k.')
    ax.set_xlim([0,75])
    ax.set_ylim([-20,-9])
if __name__ == '__main__':
    ## Read the selection function data file for all DR3 plates
#    S0 = np.genfromtxt(
#        'Selection_plates.csv',           # file name
#        skip_header=0,          # lines to skip at the top
#        skip_footer=0,          # lines to skip at the bottom
#        delimiter=',',          # column delimiter
#        dtype='float32',        # data type
#        filling_values=0)       # fill missing values with 0
#    plateid = S0[:,0]
#    dK = 0.25
#    dJK = 0.1
#    Kgrid = np.arange(0,15+dK,dK)
#    JKgrid = np.arange(-0.5,4+dJK,dJK)
#    
#    # read DR3 data
#    D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3 = readDR3()
#    
#    # halo RGB sample
#    ind_hRGB = (D>0) & (dr3.M_K50<-3.5) & (dr3.feh<-1) & (dr3.RGB>0)
#    D_hRGB, Dlow_hRGB, Dup_hRGB, X_hRGB, Y_hRGB, Z_hRGB, R_hRGB, \
#        r_hRGB, K_hRGB, JK_hRGB, plateserial_hRGB = getHaloRGB(D,\
#        Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial,ind_hRGB)
#    
#        # derive nu for haloRGB sample
    dD=0.01
    Dgrid = np.arange(0,200+dD,dD)
    nu_hRGB = nulall(S0,K_hRGB,JK_hRGB,D_hRGB, Dlow_hRGB,Dup_hRGB,\
                 plateserial_hRGB, Kgrid, JKgrid, dK, dJK, Dgrid)
    
    # save the nu value and test with RZ map
    a = np.array([dr3.obsid[ind_hRGB].reshape((len(D_hRGB))),\
                  D_hRGB,JK_hRGB, K_hRGB,np.log(nu_hRGB)]).T
    #print a,np.shape(a)
    np.savetxt('hRGB_nu.csv',a,delimiter=",")
    test_nu(R_hRGB, Z_hRGB, nu_hRGB)