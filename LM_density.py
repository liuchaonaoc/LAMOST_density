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


def readDR3(k):
   # hdulist = fits.open('/Users/cliu/mw/lamost_regular/data/DR3/DR3_b1_indspv1_photo_dist_short2.fits')
    hdulist = fits.open('/Users/cliu/mw/lamost_regular/data/DR3/DR3_KGiants_short.fits')
    dr3 = hdulist[1].data
    # XYZ
    R0=8000 #pc the distance from the Sun to the GC
    Z0=27 # height of the Sun above mid-plane
    D = dr3.distK50_RJCE/1000.0 #kpc
    Dlow = D-dr3.distK85_RJCE/1000.0 #kpc
    Dup = dr3.distK15_RJCE/1000.0-D #kpc
    D = D*k
    X,Y,Z = gal2cart(dr3.glon,dr3.glat,D*1000.0,R0,Z0) #R0=8000
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

def getPop(D,Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, ind_pop): 
    # halo RGB for XU et al.
    
    D_0 = D[ind_pop]
    Dlow_0 = Dlow[ind_pop]
    Dup_0 = Dup[ind_pop]
    X_0 = X[ind_pop]
    Y_0 = Y[ind_pop]
    Z_0 = Z[ind_pop]
    R_0 = R[ind_pop]
    r_0 = r_gc[ind_pop]
    K_0 = K[ind_pop]
    JK_0 = JK[ind_pop]
    plateserial_0 = plateserial[ind_pop]

    return D_0, Dlow_0, Dup_0, X_0, Y_0, \
        Z_0, R_0, r_0, K_0, JK_0, plateserial_0
        
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
    nu_sp = np.zeros(np.shape(Dgrid))
    for i in range(N):
        pp1 = np.exp(-(D[i]-Dgrid)**2/(2*Dlow[i]**2))
        pp2 = np.exp(-(D[i]-Dgrid)**2/(2*Dup[i]**2))
        pp = pp1
        pp[Dgrid>D[i]] = pp2[Dgrid>D[i]];
        nu00= (pp/np.sum(pp)/(Dgrid**2))*dm*dc
        nu0 = nu00*S[i]
        #print np.shape(pp),np.shape(S),np.shape(nu+nu0.reshape(np.shape(nu)))
        nu = nu+nu0#nu0.reshape(np.shape(nu));
        nu_sp = nu_sp+nu00
    return nu,nu_sp
                
# calculate stellar density for all lines of sight
def nulall(S0,K,JK,D, Dlow,Dup,plateserial, Kgrid, JKgrid, dK, dJK, Dgrid):
    start = time.clock()
    nu_i = np.zeros(np.shape(D))
    nusp_i = np.zeros(np.shape(D))
    Nplate = np.shape(S0)
    for i in range(Nplate[0]):
        #indP = plates.pid==i
        pid = S0[i,0]
        indStar = (plateserial==pid) & (D>0) & (D<Dgrid[-1])
        #print np.sum(indStar)
        if np.sum(indStar)>0:
            nu1,nu1_sp = nu_los(K[indStar],JK[indStar],\
                D[indStar],\
                Dlow[indStar],Dup[indStar],\
                S0[i,1:],Dgrid,Kgrid,JKgrid,dK,dJK)
            indnu = (~np.isinf(nu1)) & (nu1>0) & (~np.isnan(nu1))
            if np.sum(indnu)>2:
                #iD=np.array([np.int(ii) for ii in np.round((D[indStar]-Dgrid[0])/0.01) ])
                #nu_i[indStar]=nu1[iD]
                f = interp1d(Dgrid[indnu],nu1[indnu],bounds_error=False,fill_value=0)
                nu_i[indStar] = f(D[indStar])
            indnu = (~np.isinf(nu1_sp)) & (nu1_sp>0) & (~np.isnan(nu1_sp))
            if np.sum(indnu)>2:
                #iD=np.array([np.int(ii) for ii in np.round((D[indStar]-Dgrid[0])/0.01) ])
                #nu_i[indStar]=nu1[iD]
                f = interp1d(Dgrid[indnu],nu1_sp[indnu],bounds_error=False,fill_value=0)
                nusp_i[indStar] = f(D[indStar])
            #print i
    print 'Time=%(t).8f' % {'t':time.clock()-start}
    return nu_i,nusp_i

# calculate mean stellar density for all lines of sight
def nulall_pltmean(S0,K,JK,D, Dlow,Dup,plateserial, Kgrid, JKgrid, dK, dJK, Dgrid):
    start = time.clock()
    Nplate = np.shape(S0)
    nu_p = np.zeros((Nplate[0],len(Dgrid)))
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
            nu_p[i,:] = nu1
            print i
    print 'Time=%(t).8f' % {'t':time.clock()-start}
    return nu_p
 
def save_file(dr3,ind,D,Dlow,Dup,Z,R,r,lnnu,filename):
     ####
    # save the nu value and test with RZ map
    a = np.array([dr3.obsid[ind].reshape((len(D))),\
                  dr3.ra[ind].reshape((len(D))),\
                  dr3.dec[ind].reshape((len(D))),\
                  dr3.glon[ind].reshape((len(D))),\
                  dr3.glat[ind].reshape((len(D))),\
                  dr3.teff[ind].reshape((len(D))),\
                  dr3.logg[ind].reshape((len(D))),\
                  dr3.feh[ind].reshape((len(D))),\
                  dr3.rv[ind].reshape((len(D))),\
                  dr3.EW_Mgb[ind].reshape(len(D)),\
                  dr3.M_K50[ind].reshape((len(D))),\
                  dr3.M_K50[ind].reshape((len(D)))-dr3.M_K15[ind].reshape((len(D))),\
                  dr3.M_K85[ind].reshape((len(D)))-dr3.M_K50[ind].reshape((len(D))),\
                  dr3.AK_RJCE[ind].reshape((len(D))),\
                  dr3.Kmag_2mass[ind].reshape((len(D))),\
                  D,Dlow, Dup, Z, R, r,lnnu]).T
    #print a,np.shape(a)
    np.savetxt(filename,a,fmt='%d %.5f %+.5f %.5f %+.5f %.0f %.2f %.2f %.0f %.5f %.2f %.2f %.2f %.3f %.3f %.2f %.2f %.2f %.2f %.2f %.2f %.2f',\
               delimiter='', header='obsid ra dec l b teff logg feh rv Mgb MK MKerr_low MKerr_up AK K dist disterr_low disterr_up Z R r_gc lnnu')


    

def distS(ra1,dec1,ra2,dec2):
    return np.sqrt((ra1-ra2)**2*np.cos(dec1*np.pi/180.)**2+(dec1-dec2)**2)*3600. #in arcsec
    
def duplicateCompare(ra, dec, D, nu):
    '''
    Compare lnnu for duplicated  stars, this can give an assessment of the performance
    of density determination
    '''
    meannu = np.zeros(np.shape(nu))
    devnu = np.zeros(np.shape(nu))
    meanD = np.zeros(np.shape(nu))
    devD = np.zeros(np.shape(nu))
    for i in range(len(ra)):
        d = distS(ra[i],dec[i], ra, dec)
        ind0 = (d<5)# & (d>0)
        if np.sum(ind0)>0:
            meannu[i] = np.mean(nu[ind0])
            meanD[i] = np.mean(D[ind0])
            if np.sum(ind0)>1:
                devnu[i] = np.std(nu[ind0]-meannu[i])#*np.sqrt(np.sum(ind0))
                devD[i] = np.std(D[ind0]-meanD[i])#*np.sqrt(np.sum(ind0))
            #else:
            #    devnu[i] = np.abs(nu[ind0]-meannu[i])*np.sqrt(2)
            #    devD[i] = np.abs(D[ind0]-meanD[i])*np.sqrt(2)
    return meannu, devnu, meanD, devD

    
def complete(D, MK, filename):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.plot(D,MK,'k.')
    ax.set_xlim((0,100))
    ax.set_ylim((-7,5))
    plt.xlabel('Distance (kpc)',fontsize=14)
    plt.ylabel(r'$M_K$ (mag)')
    fig.show()
    
    fig.savefig(filename,bbox_inches='tight')