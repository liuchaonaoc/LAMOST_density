#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 11:01:23 2016

@author: chaoliu
"""

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#import astropy.io.fits as fits
#from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
#import time
import LM_density as lm
import emcee
import corner
from matplotlib import cm
from matplotlib.colors import Normalize

# Load LAMOST data file
#from LM_density import readDR3
#from LM_density import getHaloRGB
#import matplotlib
#matplotlib.rc('xtick', labelsize=14) 
#matplotlib.rc('ytick', labelsize=14) 

def draw_samples(samples):
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    ax.plot(np.abs(samples[:,2]),np.abs(samples[:,3]),'k.')
    ax.set_xlim([0,5])
    ax.set_ylim([0,10])
    fig.show()
    
def lnSigma_model(Rcenter, n1, n2, h):
    S = (n1)*((1-n2)*np.exp(-Rcenter/h)+(n2)*Rcenter**(-np.abs(2.5)))
    #print S
    S[(S<=1e-8) | np.isnan(S) | np.isinf(S)] = 1e-50
    lnS = np.log(S)
    return lnS
    
                  
                  
def lnprob(x, Rcenter, Sigma, Sigmaerr):
    #print x
    if x[0]>0 and x[1]>0 and\
        x[0]<1 and x[1]<1 and\
        x[2]>0.0 and x[2]<5. and x[3]>0 and x[3]<5:
        lnL = -np.sum(((Sigma)-\
                       np.exp(lnSigma_model(Rcenter,x[0],x[1],x[2],x[3])))**2/\
                       Sigmaerr**2)
    else:
        lnL=-np.inf
    
    return lnL
    
def fit_SigmaR_MCMC(Rcenter, Sigma, Sigmaerr):
    ndim = 4
    nwalkers =100
    # Choose an initial set of positions for the walkers.
    p0=np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.rand(nwalkers)*0.5+0.25
    p0[:,1] = np.random.rand(nwalkers)*0.15-0.05
    p0[:,2] = np.random.rand(nwalkers)*1.5+1.5
    p0[:,3] = np.random.rand(nwalkers)*0.5+2
    
    
    # Initialize the sampler with the chosen specs.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,\
                                    args=[Rcenter, Sigma, Sigmaerr])
    
    # Run 100 steps as a burn-in.
    pos, prob, state = sampler.run_mcmc(p0, 3000)
    
    # Reset the chain to remove the burn-in samples.
    sampler.reset()
    
    ##%% Starting from the final position in the burn-in chain, sample for 1000
    # steps.
    sampler.run_mcmc(pos, 3000, rstate0=state)
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    x = np.median(samples,axis=0)
    
    xerr = np.var(samples,axis=0)
    
    fig = corner.corner(samples, labels=["n1","n2","h","a"])
    
    return x, xerr, samples

def draw_SigmaR0(Sigma1, Sigmaerr1, Sigma2, Sigmaerr2, Sigma3, Sigmaerr3, \
                  Sigma4, Sigmaerr4, Sigma5, Sigmaerr5, \
                 popt1,perr1,popt2,perr2,popt3,perr3,\
                 popt4,perr4,popt5,perr5,Rcenter):
    #indS = (Sigma>0) & (~np.isnan(Sigma)) & (~np.isinf(Sigma))
    #print Sigma
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    e1 = ax.errorbar(Rcenter,np.log(Sigma1),yerr=Sigmaerr1/Sigma1,\
                fmt='ko',markerfacecolor='none')
    e2 = ax.errorbar(Rcenter,np.log(Sigma2),yerr=Sigmaerr2/Sigma2,\
                fmt='g+',markerfacecolor='g')
    e3 = ax.errorbar(Rcenter,np.log(Sigma3),yerr=Sigmaerr3/Sigma3,\
                fmt='b^',markerfacecolor='none',markeredgecolor='b')
    e4 = ax.errorbar(Rcenter,np.log(Sigma4),yerr=Sigmaerr4/Sigma4,\
                fmt='rx',markeredgecolor='r')
    e5 = ax.errorbar(Rcenter,np.log(Sigma5),yerr=Sigmaerr5/Sigma5,\
                fmt='ks',markerfacecolor='none')
    #Sigma_halo = Rcenter**(-popt[3])*popt[1]*popt[0]
    Sigma_disk1 = np.exp(-Rcenter/popt1[2])*popt1[0]*(1-popt1[1])
    ax.plot(Rcenter,np.log(Sigma_disk1),'k--',linewidth=1)
    Sigma_disk2 = np.exp(-Rcenter/popt2[2])*popt2[0]*(1-popt2[1])
    ax.plot(Rcenter,np.log(Sigma_disk2),'g--',linewidth=1)
    Sigma_disk3 = np.exp(-Rcenter/popt3[2])*popt3[0]*(1-popt3[1])
    ax.plot(Rcenter,np.log(Sigma_disk3),'b--',linewidth=1)
    Sigma_disk4 = np.exp(-Rcenter/popt4[2])*popt4[0]*(1-popt4[1])
    ax.plot(Rcenter,np.log(Sigma_disk4),'r:',linewidth=1)
    Sigma_disk5 = np.exp(-Rcenter/popt5[2])*popt5[0]*(1-popt5[1])
    ax.plot(Rcenter,np.log(Sigma_disk5),'k:',linewidth=1)
    plt.legend([e1,e2,e3,e4,e5],['|z|<40 kpc','0.2<|z|<40 kpc',\
               '0.4<|z|<40 kpc','0.6<|z|<40 kpc','0.8<|z|<40 kpc'])
    ax.set_xlim((7,30))
    ax.set_ylim((-12.5,-5))
    ax.set_xlabel(r'$R$ (kpc)',fontsize=14)
    ax.set_ylabel(r'$\ln\Sigma$ (pc$^{-2}$)',fontsize=14)
    fig.show()
    #fig.savefig(filename2,bbox_inches='tight')
    
def draw_SigmaR(Sigma, Sigmaerr, Rcenter, filename2):
    indS = (Sigma>0) & (~np.isnan(Sigma)) & (~np.isinf(Sigma))
    popt,pcov0 = curve_fit(lnSigma_model,Rcenter[indS],np.log(Sigma[indS]),\
                          p0=[0.3,0.15,3])
    pcov = np.sqrt(pcov0.diagonal())
#    popt,pcov,samples1 = fit_SigmaR_MCMC(Rcenter[indS],Sigma[indS],\
#                          Sigmaerr[indS])  

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
    ax.text(19,-8,r'$\Sigma_D\sim\exp(-\frac{R}{%(h).1f\pm %(ne).1f})$' %\
        {'h':popt[2],'ne':(pcov[2])},fontsize=14,color='r')
    ax.text(19,-9,r'$\Sigma_H\sim R^{-%(a).1f\pm %(ae).1f}$' %\
        {'a':2.5,'ae':(0)},fontsize=14,color='b')
    
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
    return popt,(pcov)#np.sqrt(pcov.diagonal())
    
def nu_Sigma(R,Z,nu,zrange,filename2):
    dR=1.
    dZ=zrange[1]-zrange[0]
    Rgrid = np.arange(8.,30.,dR)
    Zgrid = zrange
#    Zgrid1 = np.arange(0.,4.,dZ)
#    Zgrid2 = np.arange(4.,40.,dZ)
    Rcenter = (Rgrid[0:len(Rgrid)-1]+Rgrid[1:])/2.
    Zcenter = (Zgrid[0:len(Zgrid)-1]+Zgrid[1:])/2.
#    Zcenter1 = (Zgrid1[0:len(Zgrid1)-1]+Zgrid1[1:])/2.
#    Zcenter2 = (Zgrid2[0:len(Zgrid2)-1]+Zgrid2[1:])/2.
    #print Rcenter
    Sigma = np.zeros(np.shape(Rcenter))
    Sigmaerr = np.zeros(np.shape(Rcenter))
#    Sigma1 = np.zeros(np.shape(Rcenter))
#    Sigma1err = np.zeros(np.shape(Rcenter))
#    Sigma2 = np.zeros(np.shape(Rcenter))
#    Sigma2err = np.zeros(np.shape(Rcenter))
    nuRz = np.zeros((len(Zcenter),len(Rcenter)))
    nuRzerr = np.zeros((len(Zcenter),len(Rcenter)))
    N_Rz = np.zeros((len(Zcenter),len(Rcenter)))
    
#    iR = np.array([np.int(i) for i in np.round((R-Rcenter[0])/dR)])
#    iZ = np.array([np.int(i) for i in np.round((np.abs(Z)-Zcenter[0])/dZ)])
#    ind = (iR>=0) & (iR<len(Rcenter)) & (iZ>=0) & (iZ<len(Zcenter))
#    Rzmap = np.zeros((len(Rcenter),len(Zcenter)))
#    NRzmap = np.zeros((len(Rcenter),len(Zcenter)))
#    for i in range(len(R)):
#        if ind[i]>0 and ~np.isnan(nu[i]) and ~np.isinf(nu[i]) and nu[i]>0:
#            Rzmap[iR[i],iZ[i]] = Rzmap[iR[i],iZ[i]]+nu[i]
#            NRzmap[iR[i],iZ[i]] = NRzmap[iR[i],iZ[i]]+1
#    Rzmap = Rzmap/NRzmap  
              
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
    #Sigmaerr2 = np.zeros(np.shape(Rcenter))                
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
        print m
    for j in range(len(Rcenter)):
        Sigmaerr[j] = np.sqrt(Sigmaerr[j]**2+\
                ((np.percentile(Sigma_b[Sigma_b[:,j]>0,j],95.) -\
                            np.percentile(Sigma_b[Sigma_b[:,j]>0,j],5.))/2.)**2)
                     
#    for j in range(len(Rcenter)):
#        nuZ = np.zeros(np.shape(Zcenter1))
#        nuZerr = np.zeros(np.shape(Zcenter1))
#        for i in range(len(Zcenter1)):
#            ind = (np.abs(Z)>=Zgrid1[i]) & (np.abs(Z)<=Zgrid1[i+1]) &\
#            (nu>0) & (R>=Rgrid[j]) & (~np.isnan(nu)) & (~np.isinf(nu)) &\
#            (R<=Rgrid[j+1])
#            if np.sum(ind)>1:
#                nuZ[i] = np.median(nu[ind])
#                nuZerr[i] = np.std(nu[ind])
#        ind = (nuZ>0)
#        if np.sum(ind)>1:
#            nuZ1 = np.interp(Zcenter1,Zcenter1[ind],nuZ[ind])
#            nuZ1err = nuZerr[ind] #np.interp(Zcenter1,Zcenter1[ind],nuZerr[ind])
#            Sigma1[j] = np.sum(nuZ1)*dZ
#            Sigma1err[j] = np.sqrt(np.sum(nuZ1err**2))*dZ
##        #halo
#        nuZ = np.zeros(np.shape(Zcenter2))
#        nuZerr = np.zeros(np.shape(Zcenter2))
#        for i in range(len(Zcenter2)):
#            ind = (np.abs(Z)>=Zgrid2[i]) & (np.abs(Z)<=Zgrid2[i+1]) &\
#            (nu>0) & (R>=Rgrid[j]) & (~np.isnan(nu)) & (~np.isinf(nu)) &\
#            (R<=Rgrid[j+1])
#            if np.sum(ind)>1:
#                nuZ[i] = np.median(nu[ind])
#                nuZerr[i] = np.std(nu[ind])
#        ind = (nuZ>0)
#        if np.sum(ind)>1:
#            nuZ1 = np.interp(Zcenter2,Zcenter2[ind],nuZ[ind])
#            nuZ1err = nuZerr[ind] #np.interp(Zcenter2,Zcenter2[ind],nuZerr[ind])
#            Sigma2[j] = np.sum(nuZ1)*dZ
#            Sigma2err[j] = np.sqrt(np.sum(nuZ1err**2))*dZ
#    Sigma = Sigma1+Sigma2
#    Sigmaerr = np.sqrt(Sigma1err**2+Sigma2err**2)

    draw_SigmaR(Sigma, Sigmaerr, Rcenter, filename2)
    return Sigma, Sigmaerr, Rcenter, Sigma_b, nuRz, nuRzerr, Zcenter, \
        N_Rz#, Sigma1, Sigma1err, Sigma2, Sigma2err,

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

def draw_SigmaZ_Zslice(Rcenter, Zcenter, nuRz, Sigma, zlim,dZ, filename):
    norm = Normalize(vmin=0,vmax=5)
    cmap = cm.get_cmap('jet')
    scalar_map = cm.ScalarMappable(\
        norm=norm,
        cmap=cmap)
    scalar_map.set_array(np.array(range(zlim[0],zlim[1])))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Rcenter, np.log(Sigma),'k-',linewidth=2)
    #ax.plot(Rcenter, np.log(np.sum(nuRz,axis=0)*dZ),'r--',linewidth=2)
    for i in range(np.int((zlim[1]-zlim[0])/dZ)):
        ax.plot(Rcenter,np.log(nuRz[i,:]*dZ),'-',\
            color=scalar_map.to_rgba(Zcenter[i]))
    cb = plt.colorbar(scalar_map, cax=None, ax=ax, ticks=range(0,6))
    cb.ax.set_yticklabels([p for p in range(0, 6)])
    cb.set_label(r'|z|',fontsize=12)
    ax.set_xlabel(r'$R$ (kpc)',fontsize=12)
    ax.set_ylabel(r'$\Sigma$ (pc$^{-2}$)',fontsize=12)
    
    fig.show()
    fig.savefig(filename,bbox_inches='tight')

if __name__ == '__main__':
#    # Read the selection function data file for all DR3 plates
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
#    ###########################################################################
#    ##### disk 1
#    ### For Wang et al. 2017
#    ###########################################################################
#    # read DR3 data
#    D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3 = lm.readDR3(1.)
#    
#    # halo RGB sample
#    ind_dRGB = (K<=14.3) & (dr3.RGBdisk_WHF==84)
#    
#    D_dRGB, Dlow_dRGB, Dup_dRGB, X_dRGB, Y_dRGB, Z_dRGB, R_dRGB, \
#        r_dRGB, K_dRGB, JK_dRGB, plateserial_dRGB = lm.getPop(D,\
#        Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial,ind_dRGB)
#    
#        # derive nu for haloRGB sample
#    dD=0.01
#    Dgrid = np.arange(0,200+dD,dD)
#    nu_dRGB = lm.nulall(S0,K_dRGB,JK_dRGB,D_dRGB, Dlow_dRGB,Dup_dRGB,\
#                 plateserial_dRGB, Kgrid, JKgrid, dK, dJK, Dgrid)
#    
#    lm.save_file(dr3,ind_dRGB,D_dRGB,Dlow_dRGB,Dup_dRGB,Z_dRGB,\
#             R_dRGB,r_dRGB,np.log(nu_dRGB),'LMDR3_diskRGB.dat')
#    
#    #draw_diskRZ(R_dRGB, Z_dRGB, nu_dRGB, 'WHF_nuRZ.eps')
#    #lm.complete(D_dRGB, dr3.M_K50[ind_dRGB],'dRGB0_complete.eps')
#    
    ###########################################################################
    ##### disk 2
    ### For Liu et al. 2017
    ###########################################################################    
    #### another sample
#    D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3 = lm.readDR3(1.)
#    ind_dRGB2 = (K<=14.3) & (D>0.5) & (dr3.M_K50<-3.5) &\
#             (dr3.RGB_new==84)
#    D_dRGB2, Dlow_dRGB2, Dup_dRGB2, X_dRGB2, Y_dRGB2, Z_dRGB2, R_dRGB2, \
#        r_dRGB2, K_dRGB2, JK_dRGB2, plateserial_dRGB2 = lm.getPop(D,\
#        Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial,ind_dRGB2)
#    
#        # derive nu for haloRGB sample
#    dD=0.01
#    Dgrid = np.arange(0,200+dD,dD)
#    nu_dRGB2 = lm.nulall(S0,K_dRGB2,JK_dRGB2,D_dRGB2, Dlow_dRGB2,Dup_dRGB2,\
#                 plateserial_dRGB2, Kgrid, JKgrid, dK, dJK, Dgrid)
#    
#    lm.save_file(dr3,ind_dRGB2,D_dRGB2,Dlow_dRGB2,Dup_dRGB2,Z_dRGB2,\
#             R_dRGB2,r_dRGB2,np.log(nu_dRGB2),'LMDR3_diskRGB2.dat')
#    
#    draw_diskRZ(R_dRGB2, Z_dRGB2, nu_dRGB2, 'dRGB2_nuRZ.eps')
    dZ = 0.1
#    Sigma1,Sigmaerr1, Rcenter1, Sigma_b, nuRz1, nuRzerr1, Zcenter, N_Rz = \
#        nu_Sigma(R_dRGB2,Z_dRGB2,nu_dRGB2,np.arange(0,40.,dZ),'dRGB2_SigmaR_z0_40.eps') 

#    Sigma2,Sigmaerr2, Rcenter2, Sigma_b, nuRz, nuRzerr, Zcenter, N_Rz = \
#        nu_Sigma(R_dRGB2,Z_dRGB2,nu_dRGB2,np.arange(0.2,40.,dZ),'dRGB2_SigmaR_z0.2_40.eps')  
#
#    Sigma3,Sigmaerr3, Rcenter3, Sigma_b, nuRz, nuRzerr, Zcenter, N_Rz = \
#        nu_Sigma(R_dRGB2,Z_dRGB2,nu_dRGB2,np.arange(0.4,40.,dZ),'dRGB2_SigmaR_z0.4_40.eps')  
#    
#    Sigma4,Sigmaerr4, Rcenter4, Sigma_b, nuRz, nuRzerr, Zcenter, N_Rz = \
#        nu_Sigma(R_dRGB2,Z_dRGB2,nu_dRGB2,np.arange(0.6,40.,dZ),'dRGB2_SigmaR_z0.6_40.eps')  
#
#    Sigma5,Sigmaerr5, Rcenter5, Sigma_b, nuRz, nuRzerr, Zcenter, N_Rz = \
#        nu_Sigma(R_dRGB2,Z_dRGB2,nu_dRGB2,np.arange(0.8,40.,dZ),'dRGB2_SigmaR_z0.8_40.eps')  
# 
    
#    popt1,perr1 = draw_SigmaR(Sigma1, Sigmaerr1, Rcenter1,'dRGB2_SigmaR_z0_40.eps')
#    popt2,perr2 = draw_SigmaR(Sigma2, Sigmaerr2, Rcenter2,'dRGB2_SigmaR_z0.2_40.eps')
#    popt3,perr3 = draw_SigmaR(Sigma3, Sigmaerr3, Rcenter3,'dRGB2_SigmaR_z0.4_40.eps')
#    popt4,perr4 = draw_SigmaR(Sigma4, Sigmaerr4, Rcenter4,'dRGB2_SigmaR_z0.6_40.eps')
#    popt5,perr5 = draw_SigmaR(Sigma5, Sigmaerr5, Rcenter5,'dRGB2_SigmaR_z0.6_40.eps')
#    
#    draw_SigmaR0(Sigma1, Sigmaerr1,Sigma2, Sigmaerr2,Sigma3, Sigmaerr3,\
#                 Sigma4, Sigmaerr4,Sigma5, Sigmaerr5,\
#                 popt1,perr1,popt2,perr2,popt3,perr3,\
#                 popt4,perr4,popt5,perr5,Rcenter1)
    
#    test_dupData(dr3,ind_dRGB2,nu_dRGB2, D)
#    lm.complete(D_dRGB2, dr3.M_K50[ind_dRGB2],'dRGB2_complete.eps')
#    print np.sum(ind_dRGB2)
#    draw_diskRZ(R_dRGB2, Z_dRGB2, nu_dRGB2, 'dRGB2_nuRZ0_40.eps')


#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.imshow((nuRz1-Rzmap.T)/nuRz1, vmin=-1,vmax=1,interpolation='nearest',\
#              extent=[Rcenter1[0],Rcenter1[-1],Zcenter[-1],Zcenter[0]])
#    ax.set_xlim(Rcenter1[0],Rcenter1[-1])
#    ax.set_ylim([0,5])
#    fig.show()
    draw_SigmaZ_Zslice(Rcenter1, Zcenter, nuRz1, Sigma1, [0,5], dZ, 'Sigma_Zslice.eps')
