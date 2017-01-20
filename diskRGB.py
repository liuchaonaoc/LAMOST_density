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
def lnSigma_model(Rcenter, n1, n2, h, a):
    return np.log(n1*np.exp(-Rcenter/h)+n2*Rcenter**(-a))
    
    
def nu_Sigma(R,Z,nu,filename2):
    dR=1.
    dZ=0.25
    Rgrid = np.arange(8.,30.,dR)
    Zgrid = np.arange(0.,30.,dZ)
    Zgrid1 = np.arange(0.,4.,dZ)
    Zgrid2 = np.arange(4.,40.,dZ)
    Rcenter = (Rgrid[0:len(Rgrid)-1]+Rgrid[1:])/2.
    Zcenter = (Zgrid[0:len(Zgrid)-1]+Zgrid[1:])/2.
    Zcenter1 = (Zgrid1[0:len(Zgrid1)-1]+Zgrid1[1:])/2.
    Zcenter2 = (Zgrid2[0:len(Zgrid2)-1]+Zgrid2[1:])/2.
    #print Rcenter
    Sigma1 = np.zeros(np.shape(Rcenter))
    Sigma1err = np.zeros(np.shape(Rcenter))
    Sigma2 = np.zeros(np.shape(Rcenter))
    Sigma2err = np.zeros(np.shape(Rcenter))
    for j in range(len(Rcenter)):
        nuZ = np.zeros(np.shape(Zcenter1))
        nuZerr = np.zeros(np.shape(Zcenter1))
        for i in range(len(Zcenter1)):
            ind = (np.abs(Z)>=Zgrid1[i]) & (np.abs(Z)<=Zgrid1[i+1]) &\
            (nu>0) & (R>=Rgrid[j]) & (~np.isnan(nu)) & (~np.isinf(nu)) &\
            (R<=Rgrid[j+1])
            if np.sum(ind)>1:
                nuZ[i] = np.median(nu[ind])
                nuZerr[i] = np.std(nu[ind])
        ind = (nuZ>0)
        if np.sum(ind)>5:
            nuZ1 = np.interp(Zcenter1,Zcenter1[ind],nuZ[ind])
            nuZ1err = np.interp(Zcenter1,Zcenter1[ind],nuZerr[ind])
            Sigma1[j] = np.sum(nuZ1)*dZ
            Sigma1err[j] = np.sqrt(np.sum(nuZ1err**2))*dZ
        #halo
        nuZ = np.zeros(np.shape(Zcenter2))
        nuZerr = np.zeros(np.shape(Zcenter2))
        for i in range(len(Zcenter2)):
            ind = (np.abs(Z)>=Zgrid2[i]) & (np.abs(Z)<=Zgrid2[i+1]) &\
            (nu>0) & (R>=Rgrid[j]) & (~np.isnan(nu)) & (~np.isinf(nu)) &\
            (R<=Rgrid[j+1])
            if np.sum(ind)>1:
                nuZ[i] = np.median(nu[ind])
                nuZerr[i] = np.std(nu[ind])
        ind = (nuZ>0)
        if np.sum(ind)>5:
            nuZ1 = np.interp(Zcenter2,Zcenter2[ind],nuZ[ind])
            nuZ1err = np.interp(Zcenter2,Zcenter2[ind],nuZerr[ind])
            Sigma2[j] = np.sum(nuZ1)*dZ
            Sigma2err[j] = np.sqrt(np.sum(nuZ1err**2))*dZ
    Sigma = Sigma1+Sigma2
    Sigmaerr = np.sqrt(Sigma1err**2+Sigma2err**2)
    indS = (Sigma>0) & (~np.isnan(Sigma)) & (~np.isinf(Sigma))
    popt,pcov = curve_fit(lnSigma_model,Rcenter[indS],np.log(Sigma[indS]),p0=[0.3,0.15,1.8,3])
    Sigma_halo = Rcenter**(-popt[3])*popt[1]
    Sigma_disk = np.exp(-Rcenter/popt[2])*popt[0]
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
    ax.text(19,-8,r'$\Sigma_D\sim\exp(-\frac{R}{%(h).1f\pm%(ne).1f})$' %\
        {'h':popt[2],'ne':np.sqrt(pcov[2,2])},fontsize=14,color='r')
    ax.text(19,-9,r'$\Sigma_H\sim R^{-%(a).1f\pm%(ae).1f}$' %\
        {'a':popt[3],'ae':np.sqrt(pcov[3,3])},fontsize=14,color='b')
    
    ax2.set_ylabel(r'$\Sigma_H/(\Sigma_D+\Sigma_H)$',fontsize=14)
    ax2.plot(Rcenter,Sigma_halo/(Sigma_disk+Sigma_halo),'k-')
    ax2.set_ylim((0,1))
    print Rcenter,Sigma_halo/(Sigma_disk+Sigma_halo)
    ax.set_xlim((7,30))
    ax.set_ylim((-12.5,-5))
    ax.set_xlabel(r'$R$ (kpc)',fontsize=14)
    ax.set_ylabel(r'$\ln\Sigma$ (pc$^{-2}$)',fontsize=14)
    fig.show()
    fig.savefig(filename2,bbox_inches='tight')
    return Sigma

def draw_diskRZ(R,Z,nu,filename1):
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
    ax.plot([0.,30.],[0.,0.],'--k')
    #plt.clabel(CS,[-12,-11,-10,-9,-8,-7,-6],fmt='%.0f',fontsize=6)
    ax.set_ylim([-15,15])
    ax.set_xlim([0,25])
    plt.xlabel(r'$R$ (kpc)',fontsize=14)
    plt.ylabel(r'$Z$ (kpc)',fontsize=14)
    fig.show()
    fig.savefig(filename1,bbox_inches='tight')
    

#    indRz = (np.isnan(Rzmap)) | (np.isinf(Rzmap))
#    Rzmap[indRz] = 0.
#    Sigma = np.max(Rzmap,axis=1)
#    #print Rzmap
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(Rgrid,np.log(Sigma),'k-')
#    ax.plot(Rgrid,np.log(np.exp(-Rgrid/1.7)*24),'r--')
#    ax.plot(Rgrid,np.log(Rgrid**(-3)*0.2),'b--')
#    ax.plot(Rgrid,np.log(np.exp(-Rgrid/1.7)*24+Rgrid**(-3.8)*0.2),'g--',linewidth=2)
#    ax.text(20,-2.5,r'$\Sigma_D(R)\sim\exp(-{R/{1.7}})$',fontsize=14,color='r')
#    ax.text(20,-4.5,r'$\Sigma_H(R)\sim R^{-3.8}$',fontsize=14,color='b')
#    ax.text(20,-6.5,r'$\Sigma=\Sigma_D+\Sigma_H$',fontsize=14,color='g')
#    ax.set_xlim([6,40])
#    ax.set_ylim([-17,0])
#    fig.show()
#    fig.savefig(filename2,bbox_inches='tight')

def test_dupData(dr3,ind_dRGB, nu_dRGB, D):
    #test
    meannu,devnu,meanD, devD = lm.duplicateCompare(dr3.ra[ind_dRGB], \
            dr3.dec[ind_dRGB], D[ind_dRGB], nu_dRGB)
    
    dg1 = np.arange(0,2,step=0.04)
    dc1 = (dg1[0:len(dg1)-1]+dg1[1:])/2.
    #  distribution of the relative error of nu 
    ind = (devnu>0.0) & (meannu>0.0)
    nuerr = (devnu[ind]/meannu[ind])
    h,x = np.histogram(nuerr,bins=dg1)
    popt,pcov = curve_fit(gaussian,dc1,h)
    print 'error of nu:\n sigma=%(s).3f+/=%(e).3f' % {'s':np.abs(popt[1]),'e':pcov[1,1]}
    print 'median dev=%(s).3f' % {'s':np.median(nuerr)}
    fig = plt.figure(figsize=[4,3.5])
    ax = fig.add_subplot(111)
    ax.plot(dc1,h,'k-')
    ax.plot(dc1,gaussian(dc1,popt[0],popt[1]),'r--')
    ax.text(0.5,350.,r'Gaussian $\sigma$=%(s).3f' % {'s':np.abs(popt[1])},fontsize=14)
    ax.text(0.5,300.,r'median=%(s).3f' % {'s':np.median(nuerr)},fontsize=14)
    plt.xlabel(r'$\sigma$($\nu$)/$\nu$',fontsize=14)
    plt.ylabel('Count',fontsize=14)
    fig.show()
    fig.savefig('Deltalnnu_dRGB.eps',bbox_inches='tight')
    
    #distribution of the relative error of Distance
    dg = np.arange(0,2,step=0.01)
    dc = (dg[0:len(dg)-1]+dg[1:])/2.
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
    ax.text(0.15,1050.,r'Gaussian $\sigma$=%(s).3f' % {'s':np.abs(popt[1])},fontsize=14)
    ax.text(0.15,900.,r'median=%(s).3f' % {'s':np.median(Derr)},fontsize=14)
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
    ###########################################################################
    ##### disk 1
    ### For Wang et al. 2017
    ###########################################################################
    # read DR3 data
    D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3 = lm.readDR3(1.)
    
    # halo RGB sample
    ind_dRGB = (K<=14.3) & (dr3.RGBdisk_WHF==84)
    
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
    
    #draw_diskRZ(R_dRGB, Z_dRGB, nu_dRGB, 'WHF_nuRZ.eps')
    #lm.complete(D_dRGB, dr3.M_K50[ind_dRGB],'dRGB0_complete.eps')
    
    ###########################################################################
    ##### disk 2
    ### For Liu et al. 2017
    ###########################################################################    
    #### another sample
    ind_dRGB2 = (K<=14.3) & (D>0.5) & (dr3.M_K50<-3.5) &\
             (dr3.RGB_new==84)
    D_dRGB2, Dlow_dRGB2, Dup_dRGB2, X_dRGB2, Y_dRGB2, Z_dRGB2, R_dRGB2, \
        r_dRGB2, K_dRGB2, JK_dRGB2, plateserial_dRGB2 = lm.getPop(D,\
        Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial,ind_dRGB2)
    
        # derive nu for haloRGB sample
    #dD=0.01
    #Dgrid = np.arange(0,200+dD,dD)
    nu_dRGB2 = lm.nulall(S0,K_dRGB2,JK_dRGB2,D_dRGB2, Dlow_dRGB2,Dup_dRGB2,\
                 plateserial_dRGB2, Kgrid, JKgrid, dK, dJK, Dgrid)
    
    lm.save_file(dr3,ind_dRGB2,D_dRGB2,Dlow_dRGB2,Dup_dRGB2,Z_dRGB2,\
             R_dRGB2,r_dRGB2,np.log(nu_dRGB2),'LMDR3_diskRGB2.dat')
    
    draw_diskRZ(R_dRGB2, Z_dRGB2, nu_dRGB2, 'dRGB2_nuRZ.eps')
    nu_Sigma(R_dRGB2,Z_dRGB2,nu_dRGB2,'dRGB2_SigmaR.eps')    
    test_dupData(dr3,ind_dRGB2,nu_dRGB2, D)
    lm.complete(D_dRGB2, dr3.M_K50[ind_dRGB2],'dRGB2_complete.eps')
    print np.sum(ind_dRGB2)
    
    