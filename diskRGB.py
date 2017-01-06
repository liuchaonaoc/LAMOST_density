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
from scipy.optimize import curve_fit
import time
import LM_density as lm
# Load LAMOST data file
#from LM_density import readDR3
#from LM_density import getHaloRGB

def draw_diskRZ(R,Z,nu):
    '''
    draw density map in R-Z plane to check if the lnnu makes sense
    '''
    #draw RZ map
    dR=0.2
    dZ=0.2
    Rgrid = np.arange(0,100.,dR)
    Zgrid = np.arange(-100,100.,dZ)
    iR = np.array([np.int(i) for i in np.round((R-Rgrid[0])/dR)])
    iZ = np.array([np.int(i) for i in np.round(((Z)-Zgrid[0])/dZ)])
    ind = (iR>=0) & (iR<len(Rgrid)) & (iZ>=0) & (iZ<len(Zgrid))
    Rzmap = np.zeros((len(Rgrid),len(Zgrid)))
    NRzmap = np.zeros((len(Rgrid),len(Zgrid)))
    for i in range(len(R)):
        if ind[i]>0 and ~np.isnan(nu[i]) and ~np.isinf(nu[i]) and nu[i]>0:
            Rzmap[iR[i],iZ[i]] = Rzmap[iR[i],iZ[i]]+nu[i]
            NRzmap[iR[i],iZ[i]] = NRzmap[iR[i],iZ[i]]+1
    Rzmap = Rzmap / NRzmap    
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.log(Rzmap.T),vmin=-13,vmax=-4,interpolation='nearest',\
              extent=[0,100,100,-100])
    plt.colorbar(im)
    plt.title(r'$\ln\nu$',fontsize=14)
    CS = ax.contour(Rgrid,Zgrid,np.log(Rzmap.T),[-12,-11,-10,-9,-8,-7,-6],\
               colors='k',linestyles='solid')
    th=np.arange(0,np.pi*2,step=0.01)
    for rr in np.arange(2,30,step=2):
        ax.plot(rr*np.cos(th),rr*np.sin(th),'k:')
    #plt.clabel(CS,[-12,-11,-10,-9,-8,-7,-6],fmt='%.0f',fontsize=6)
    ax.set_ylim([-15,15])
    ax.set_xlim([0,30])
    plt.xlabel(r'$R$ (kpc)',fontsize=14)
    plt.ylabel(r'$Z$ (kpc)',fontsize=14)
    fig.show()
    fig.savefig('WHF_nuRZ.eps',bbox_inches='tight')
    

    indRz = (np.isnan(Rzmap)) | (np.isinf(Rzmap))
    Rzmap[indRz] = 0.
    Sigma = np.sum(Rzmap,axis=1)
    #print Rzmap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Rgrid,np.log(Sigma),'k-')
    ax.plot(Rgrid,np.log(np.exp(-Rgrid/1.7)*24),'r--')
    ax.plot(Rgrid,np.log(Rgrid**(-3.8)*0.2),'b--')
    ax.plot(Rgrid,np.log(np.exp(-Rgrid/1.7)*24+Rgrid**(-3.8)*0.2),'g--',linewidth=2)
    ax.text(25,-4.5,r'$\Sigma_D(R)\sim\exp(-{R/{1.7}})$',fontsize=14,color='r')
    ax.text(25,-6.5,r'$\Sigma_H(R)\sim R^{-3.8}$',fontsize=14,color='b')
    ax.text(25,-8.5,r'$\Sigma=\Sigma_D+\Sigma_H$',fontsize=14,color='g')
    ax.set_xlim([6,50])
    ax.set_ylim([-20,0])
    fig.show()
    fig.savefig('WHF_SigmaR.eps',bbox_inches='tight')

def test_dupData(dr3,ind_dRGB, nu_dRGB, D):
    #test
    meannu,devnu,meanD, devD = lm.duplicateCompare(dr3.ra[ind_dRGB], \
            dr3.dec[ind_dRGB], D[ind_dRGB], nu_dRGB)
    
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
    ax.text(0.5,160,r'Gaussian $\sigma$=%(s).3f' % {'s':np.abs(popt[1])},fontsize=14)
    ax.text(0.5,140,r'median=%(s).3f' % {'s':np.median(nuerr)},fontsize=14)
    plt.xlabel(r'$\sigma$($\nu$)/$\nu$',fontsize=14)
    plt.ylabel('Count',fontsize=14)
    fig.show()
    fig.savefig('Deltalnnu_dRGB.eps',bbox_inches='tight')
    
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
    ax.text(0.15,1200,r'Gaussian $\sigma$=%(s).3f' % {'s':np.abs(popt[1])},fontsize=14)
    ax.text(0.15,1000,r'median=%(s).3f' % {'s':np.median(Derr)},fontsize=14)
    ax.set_xlim((0,0.5))
    plt.xlabel(r'$\sigma$(Dist)/Dist',fontsize=14)
    plt.ylabel('Count',fontsize=14)
    fig.show()
    fig.savefig('DeltaDist_dRGB.eps',bbox_inches='tight')
    
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
    ind_dRGB = (dr3.RGBdisk_WHF==84)
    
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
    
    draw_diskRZ(R_dRGB, Z_dRGB, nu_dRGB)
    
    test_dupData(dr3,ind_dRGB,nu_dRGB, D)
    
    