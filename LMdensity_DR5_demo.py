#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:06:52 2018

@author: cliu
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import time
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import Normalize
from LM_density_DR5 import LM_density

'''
test
'''
#%%
start = time.time()
Nu = LM_density('Selection_plates_DR5.csv')
print('escape time: %(t).0f' % {'t':time.time()-start})

#%%
catfile = 'DR3_KGiants_short_Wang18.fits'

start = time.time()
#read catalog fits file with requiredfields, including distance and errors, J and K mags from 2mass
# and plateserial, which is the serial number of plates recorded in a separated csv file.
dr3 = Nu.readCatalog(catfile,colD='distK50_RJCE',colDlow='distK85_RJCE',\
                    colDup='distK15_RJCE',unitD = 'pc',colKmag='Kmag_2mass',\
                   colJmag='Jmag_2mass',colPid='plateserial',colGlon ='glon',\
                   colGlat='glat')
#here is a chance to further select a subset of the catalog for calculating nu
ind_hRGB = dr3['distK50_RJCE']>0
#calculate nu for the selected stars
Nu.nulall(ind_hRGB)
print('escape time: %(t).0f' % {'t':time.time()-start})

#%%
'''
Draw density map in R-z plane
'''

def draw_diskRZ(R,Z,nu):
    '''
    draw density map in R-Z plane to check if the lnnu makes sense
    '''
    #draw RZ map
    dR=0.5
    dZ=0.5
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
    plt.set_cmap('jet')
    ax = fig.add_subplot(111)
    im = ax.imshow(np.log(Rzmap.T),vmin=-9,vmax=2,interpolation='nearest',\
              extent=[0,100,100,-100])
    plt.colorbar(im)
    plt.title(r'$\ln\nu$',fontsize=14)
    ax.contour(Rgrid,Zgrid,np.log(Rzmap.T),[-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2],\
               colors='k',linestyles='solid')
    th=np.arange(0,np.pi*2,step=0.01)
    for rr in np.arange(2,30,step=2):
        ax.plot(rr*np.cos(th),rr*np.sin(th),'k:')
    ax.plot([0.,30.],[0.,0.],'--k')
    #plt.clabel(CS,[-12,-11,-10,-9,-8,-7,-6],fmt='%.0f',fontsize=6)
    ax.set_ylim([-15,15])
    ax.set_xlim([0,25])
    plt.xlabel(r'$R$ (kpc)',fontsize=14)
    plt.ylabel(r'$Z$ (kpc)',fontsize=14)
    fig.show()
    #fig.savefig(filename1,bbox_inches='tight')
   
draw_diskRZ(Nu.R[ind_hRGB],Nu.Z[ind_hRGB],Nu.nu_i[ind_hRGB])


#%%
'''
Calculate surface density profile (Sigma(R))
'''
def nu_Sigma(R,Z,nu,zrange):
    '''
    At each R bin, integrate stellar volume density over a range of Z to obtain the surface density (Sigma)
    '''
    dR=1.
    dZ=zrange[1]-zrange[0]
    Rgrid = np.arange(8.,30.,dR)
    Zgrid = zrange
    Rcenter = (Rgrid[0:len(Rgrid)-1]+Rgrid[1:])/2.
    Zcenter = (Zgrid[0:len(Zgrid)-1]+Zgrid[1:])/2.
    Sigma = np.zeros(np.shape(Rcenter))
    Sigmaerr = np.zeros(np.shape(Rcenter))
    nuRz = np.zeros((len(Zcenter),len(Rcenter)))
    nuRzerr = np.zeros((len(Zcenter),len(Rcenter)))
    N_Rz = np.zeros((len(Zcenter),len(Rcenter)))
    for j in range(len(Rcenter)):
        nuZ = np.zeros(np.shape(Zcenter))
        nuZerr = np.zeros(np.shape(Zcenter))
        for i in range(len(Zcenter)):
            ind = (np.abs(Z)>=Zgrid[i]) & (np.abs(Z)<=Zgrid[i+1]) &\
                  (nu>0) & (R>=Rgrid[j]) & (~np.isnan(nu)) & \
                  (~np.isinf(nu)) & (R<=Rgrid[j+1])
            N_Rz[i,j] = np.sum(ind)
            if np.sum(ind)>1:
                nuZ[i] = np.mean(nu[ind])
                nuZerr[i] = np.std(nu[ind])
                
        ind = (nuZ>0)
        if np.sum(ind)>1:
            nuZ1 = np.interp(Zcenter,Zcenter[ind],nuZ[ind],\
                             left=max(nuZ[ind]),right=min(nuZ[ind]))
            nuRz[:,j] = nuZ1 #np.interp(Zcenter,Zcenter[ind],nuZ[ind])
            nuZ1err = np.interp(Zcenter,Zcenter[ind],nuZerr[ind])
            nuRzerr[:,j] = nuZ1err
            Sigma[j] = np.sum(nuZ1)*dZ
            Sigmaerr[j] = np.sqrt(np.sum(nuZ1err**2))*dZ
    #bootstraping for Sigma error
    N = len(nu)
    Sigma_b = np.zeros((N,len(Rcenter)))
    #number of bootrstrap
    M = 100
    
    for m in range(M):
        ind_b = np.random.choice(range(N), N, replace=True)
        R_b = R[ind_b]
        nu_b = nu[ind_b]
        Z_b = Z[ind_b]
        for j in range(len(Rcenter)):
            nuZ = np.zeros(np.shape(Zcenter))
            for i in range(len(Zcenter)):
                ind = (np.abs(Z_b)>=Zgrid[i]) & (np.abs(Z_b)<=Zgrid[i+1]) &\
                      (nu_b>0) & (R_b>=Rgrid[j]) & \
                      (~np.isnan(nu_b)) & (~np.isinf(nu_b)) &\
                      (R_b<=Rgrid[j+1])
                if np.sum(ind)>1:
                    nuZ[i] = np.mean(nu_b[ind])
                    
            ind = (nuZ>0)
            if np.sum(ind)>1:
                nuZ1 = np.interp(Zcenter,Zcenter[ind],nuZ[ind],\
                                 left=max(nuZ[ind]),right=min(nuZ[ind]))
                Sigma_b[m,j] = np.sum(nuZ1)*dZ
        print(m)
    for j in range(len(Rcenter)):
        Sigmaerr[j] = np.sqrt(Sigmaerr[j]**2+\
                ((np.percentile(Sigma_b[Sigma_b[:,j]>0,j],95.) -\
                            np.percentile(Sigma_b[Sigma_b[:,j]>0,j],5.))/2.)**2)
                     
    
    return Sigma, Sigmaerr, Rcenter, Sigma_b, nuRz, nuRzerr, Zcenter, \
        N_Rz

    
Sigma1,Sigmaerr1, Rcenter1, Sigma_b, nuRz1, nuRzerr1, Zcenter, N_Rz = \
    nu_Sigma(Nu.R[ind_hRGB],Nu.Z[ind_hRGB],Nu.nu_i[ind_hRGB],np.arange(0,40.,0.5))
    
#%%
'''
Draw stellar surface denbsity as a function of R
'''

def lnSigma_model(Rcenter, n1, n2, h):
    '''
    exponential fisc+power law halo model
    '''
    S = (n1)*((1-n2)*np.exp(-Rcenter/h)+(n2)*Rcenter**(-np.abs(2.5)))
    #print S
    S[(S<=1e-8) | np.isnan(S) | np.isinf(S)] = 1e-50
    lnS = np.log(S)
    return lnS

def draw_SigmaR(Sigma, Sigmaerr, Rcenter):
    '''
    Draw Sigma vs. R
    Sigma: stellar surface density
    Sigmaerr: error of Sigma
    Rcenter: bin in R
    '''
    indS = (Sigma>0) & (~np.isnan(Sigma)) & (~np.isinf(Sigma))
    popt,pcov0 = curve_fit(lnSigma_model,Rcenter[indS],np.log(Sigma[indS]),\
                          p0=[4,0.15,3])
    pcov = np.sqrt(pcov0.diagonal())


    Sigma_halo = Rcenter**(-2.5)*popt[1]*popt[0]
    Sigma_disk = np.exp(-Rcenter/popt[2])*popt[0]*(1-popt[1])
    #print Sigma
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.errorbar(Rcenter,np.log(Sigma),yerr=Sigmaerr/Sigma,fmt='ko',markerfacecolor='none')
    #ax.errorbar(Rcenter,np.log(Sigma1),yerr=Sigma1err/Sigma1,fmt='r+')
    #ax.plot(Rcenter,np.log(Sigma2),'bx')
    ax.plot(Rcenter,np.log(Sigma_disk),'r--',linewidth=2)
    ax.plot(Rcenter,np.log(Sigma_halo),'b--',linewidth=2)
    ax.plot(Rcenter,np.log(Sigma_halo+Sigma_disk),\
            'g-.',linewidth=2)
    ax.text(19,2.5,r'$\Sigma_D\sim\exp(-\frac{R}{%(h).1f\pm %(ne).1f})$' %\
        {'h':popt[2],'ne':(pcov[2])},fontsize=14,color='r')
    ax.text(19,1.5,r'$\Sigma_H\sim R^{-%(a).1f\pm %(ae).1f}$' %\
        {'a':2.5,'ae':(0)},fontsize=14,color='b')
    
    ax2.set_ylabel(r'$\Sigma_H/(\Sigma_D+\Sigma_H)$',fontsize=14)
    ax2.plot(Rcenter,Sigma_halo/(Sigma_disk+Sigma_halo),'k-')
    ax2.set_ylim((0,1))
    print(Rcenter,Sigma_halo/(Sigma_disk+Sigma_halo))
    ax.set_xlim((7,30))
    ax.set_ylim((-4,5))
    ax.set_xlabel(r'$R$ (kpc)',fontsize=14)
    ax.set_ylabel(r'$\ln\Sigma$ (pc$^{-2}$)',fontsize=14)
    fig.show()
    #fig.savefig(filename2,bbox_inches='tight')
    #return popt,(pcov)#np.sqrt(pcov.diagonal())
    
draw_SigmaR(Sigma1, Sigmaerr1, Rcenter1)