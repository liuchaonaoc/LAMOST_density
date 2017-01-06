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
from scipy.optimize import curve_fit
import time
import LM_density as lm
# Load LAMOST data file
#from LM_density import readDR3
#from LM_density import getHaloRGB

def straightline(x,a,b):
    return a*x+b
    
def draw_haloRZ(R,Z,nu):
    '''
    draw density map in R-Z plane to check if the lnnu makes sense
    '''
    #draw RZ map
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
    Zmesh,Rmesh = np.meshgrid(Rgrid,Zgrid)
    Rmesh = Rmesh.reshape((len(Rgrid)*len(Zgrid),))
    Zmesh = Zmesh.reshape((len(Rgrid)*len(Zgrid),))
    numesh = Rzmap.reshape((len(Rgrid)*len(Zgrid),))
    lnnumesh = np.log(numesh)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im=ax.imshow(np.log(Rzmap.T),vmin=-18,vmax=-9,interpolation='nearest',\
              extent=[0,100,100,0])
    plt.colorbar(im)
    plt.title(r'$\ln\nu$',fontsize=16)
    lnnu0 = [-15.8,-15,-14.4,-13.5,-12.5]
    for i in range(len(lnnu0)):
        indnu = np.abs(lnnumesh-lnnu0[i])<0.2
        R = Rmesh[indnu]
        Z = Zmesh[indnu]
        r = np.sqrt(R**2+Z**2)
        th = np.arctan(Z/R)
        thgrid = np.arange(5*np.pi/180.,90*np.pi/180.,step=0.15)
        rmd = np.zeros(np.shape(thgrid))
        for j in range(len(thgrid)):
            indth = np.abs(th-thgrid[j])<0.2
            if np.sum(indth)>0:
                rmd[j] = np.median(r[indth])
        indr = (rmd>0)
        Rmd = rmd[indr]*np.cos(thgrid[indr])
        Zmd = rmd[indr]*np.sin(thgrid[indr])
        ax.plot(Rmd,Zmd,'k-',linewidth=2)
        ax.text(Rmd[0]-10,25-i*4,'%(s).1f' % {'s':lnnu0[i]},fontsize=8)
    th=np.arange(0,np.pi*2,step=0.01)
    for rr in np.arange(10,100,step=10):
        ax.plot(rr*np.cos(th),rr*np.sin(th),'k:') 
    plt.xlabel(r'$R$ (kpc)',fontsize=14)
    plt.ylabel(r'$Z$ (kpc)',fontsize=14)
    ax.set_ylim([0,80])
    ax.set_xlim([0,80])
    fig.show()
    fig.savefig('HaloRGBab_nuRZ.eps',bbox_inches='tight')
    
    #draw nu vs. r
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot((np.sqrt(Rmesh**2+Zmesh**2)),np.log(numesh),'.',color=[0.5,0.5,0.5])
    rgrid = np.arange(5,80,step=0.1)
    ax.plot(rgrid, np.log(rgrid**(-2.7)*0.003),'r--',linewidth=2)
    rgrid = np.arange(10,80,step=0.1)
    ax.plot(rgrid, np.log(rgrid**(-3.9)*0.25),'b--',linewidth=2)
    ax.text(30,-10,r'$\nu\propto r^{-2.7}$',fontsize=14,color='r')
    ax.text(30,-11,r'$\nu\propto r^{-3.9}$',fontsize=14,color='b')
    ax.set_xlim([0,75])
    ax.set_ylim([-20,-9])
    plt.xlabel(r'r (kpc)',fontsize=14)
    plt.ylabel(r'$\ln\nu$ (pc$^{-3}$)',fontsize=14)
    fig.show()
    fig.savefig('HaloRGB_nur.eps',bbox_inches='tight')
    

def test_dupData(dr3,ind_hRGB, nu_hRGB, D):
    #test
    meannu,devnu,meanD, devD = lm.duplicateCompare(dr3.ra[ind_hRGB], \
            dr3.dec[ind_hRGB], D[ind_hRGB], nu_hRGB)
    
    
    dg = np.arange(0,2,step=0.01)
    dc = (dg[0:len(dg)-1]+dg[1:])/2.
    #  distribution of the relative error of nu 
    ind = (devnu>0.0) & (meannu>0.0)
    nuerr = (devnu[ind]/meannu[ind])
    h,x = np.histogram(nuerr,bins=dg)
    popt,pcov = curve_fit(gaussian,dc,h)
    print 'error of nu:\n sigma=%(s).3f+/=%(e).3f' % {'s':np.abs(popt[1]),'e':pcov[1,1]}
    print 'median dev=%(s).3f' % {'s':np.median(nuerr)}
    fig = plt.figure(figsize=[4,3.5])
    ax = fig.add_subplot(111)
    ax.plot(dc,h,'k-')
    ax.plot(dc,gaussian(dc,popt[0],popt[1]),'r--')
    ax.text(0.5,14,r'Gaussian $\sigma$=%(s).3f' % {'s':np.abs(popt[1])},fontsize=14)
    ax.text(0.5,12,r'median=%(s).3f' % {'s':np.median(nuerr)},fontsize=14)
    plt.xlabel(r'$\sigma$($\nu$)/$\nu$',fontsize=14)
    plt.ylabel('Count',fontsize=14)
    fig.show()
    fig.savefig('Deltalnnu_hRGB.eps',bbox_inches='tight')
    
    #distribution of the relative error of Distance
    ind = (devD>0.0) & (meanD>0.0)
    Derr = (devD[ind]/meanD[ind])
    h,x = np.histogram(Derr,bins=dg)
    popt,pcov = curve_fit(gaussian,dc,h)
    print 'error of distance: \n sigma=%(s).3f+/=%(e).3f' % {'s':np.abs(popt[1]),'e':pcov[1,1]}
    print 'median dev=%(s).3f' % {'s':np.median(Derr)}
    fig = plt.figure(figsize=[4,3.5])
    ax = fig.add_subplot(111)
    ax.plot(dc,h,'k-')
    ax.plot(dc,gaussian(dc,popt[0],popt[1]),'r--')
    ax.text(0.15,60,r'Gaussian $\sigma$=%(s).3f' % {'s':np.abs(popt[1])},fontsize=14)
    ax.text(0.15,50,r'median=%(s).3f' % {'s':np.median(Derr)},fontsize=14)
    ax.set_xlim((0,0.5))
    plt.xlabel(r'$\sigma$(Dist)/Dist',fontsize=14)
    plt.ylabel('Count')
    fig.show()
    fig.savefig('DeltaDist_hRGB.eps',bbox_inches='tight')
    
    
def gaussian(x,a,c):
    return a*np.exp(-(x-0)**2/(2*c**2))
   
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
    D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3 = lm.readDR3(1.)
    
    # halo RGB sample
    ind_hRGB = (D>0) & (dr3.M_K50<-3.5) & (dr3.feh<-1) & (dr3.RGBhalo_xuyan==84)
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
    
    draw_haloRZ(R_hRGB, Z_hRGB, nu_hRGB)
    
    test_dupData(dr3,ind_hRGB,nu_hRGB, D)

#### test D*0.8
#    # read DR3 data
#    D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3 = lm.readDR3(0.8)
#    
#    # halo RGB sample
#    ind_hRGB = (D>0) & (dr3.M_K50<-3.5) & (dr3.feh<-1) & (dr3.RGBhalo_xuyan==84)
#    D_hRGB, Dlow_hRGB, Dup_hRGB, X_hRGB, Y_hRGB, Z_hRGB, R_hRGB, \
#        r_hRGB, K_hRGB, JK_hRGB, plateserial_hRGB = lm.getPop(D,\
#        Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial,ind_hRGB)
#    
#    
#        # derive nu for haloRGB sample
#    dD=0.01
#    Dgrid = np.arange(0,200+dD,dD)
#    nu_hRGB85 = lm.nulall(S0,K_hRGB,JK_hRGB,D_hRGB, Dlow_hRGB,Dup_hRGB,\
#                 plateserial_hRGB, Kgrid, JKgrid, dK, dJK, Dgrid)
#    
#    lm.save_file(dr3,ind_hRGB,D_hRGB,Dlow_hRGB,Dup_hRGB,Z_hRGB,\
#             R_hRGB,r_hRGB,np.log(nu_hRGB85),'LMDR3_haloRGB_0.8D.dat')