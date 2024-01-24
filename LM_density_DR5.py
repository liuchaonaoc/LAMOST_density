# -*- coding: utf-8 -*-
"""
Spyder Editor

LAMOST density DR5
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import time

class LM_density:
    S0 = []
    plateid = []
    dK = 0.25
    dJK = 0.1
    Kgrid = np.arange(0,15+dK,dK)
    JKgrid = np.arange(-0.5,4+dJK,dJK)
    dD=0.01
    Dgrid = np.arange(0.,200.+dD,dD)
    R0=8340. #pc the distance from the Sun to the GC
    Z0=27. # height of the Sun above mid-plane
    D = []
    Dlow = []
    Dup = []
    X = []
    Y = []
    Z = []
    R = []
    r_gc = []
    K = []
    JK = []
    plateserial = []
    nu_i = []
    indPop = []
    
    def __init__(self, platefile):
        # Read the selection function data file for all DR3 plates
        self.S0 = np.genfromtxt(
            platefile,           # file name
            skip_header=0,          # lines to skip at the top
            skip_footer=0,          # lines to skip at the bottom
            delimiter=',',          # column delimiter
            dtype='float32',        # data type
            filling_values=0)       # fill missing values with 0
        self.plateid = self.S0[:,0]
        
    # gal2cart
    def gal2cart(self,l,b):
        D1 = self.D*1000.0
        cl = np.cos(l*np.pi/180.0)
        sl = np.sin(l*np.pi/180.0)
        cb = np.cos(b*np.pi/180.0)
        sb = np.sin(b*np.pi/180.0)
        x = D1*cl*cb-self.R0 #x=0 at GC; x=-Rsun at the Sun
        z = self.Z0+D1*sb
        y = D1*sl*cb #positive point to rotation direction
        return x,y,z
    
    def readCatalog(self, filename, colD='distK50_RJCE',colDlow='distK85_RJCE',\
                    colDup='distK15_RJCE',unitD = 'pc',colKmag='Kmag_2mass',\
                   colJmag='Jmag_2mass',colPid='plateserial',colGlon ='glon',\
                   colGlat='glat'):
        '''
        It can only read fiots table with flexable columns. 
        The columns that will be used in this class should be pointed out
        '''
        hdulist = fits.open(filename)
        dr3 = hdulist[1].data.copy()
        hdulist.close()
        # XYZ

        self.D = dr3[colD]
        self.Dlow = self.D-dr3[colDlow]
        self.Dup = dr3[colDup]-self.D
        if unitD=='pc':
            self.D = self.D/1000. #kpc
            self.Dlow = self.Dlow/1000.0 #kpc
            self.Dup = self.Dup/1000.0 #kpc
        #D = D*k
        #print(np.shape(self.D),np.shape(dr3[colGlon]),np.shape(dr3[colGlat]))
        #x,y,z = self.gal2cart(dr3[colGlon], dr3[colGlat]) 
        self.X,self.Y,self.Z = self.gal2cart(dr3[colGlon],dr3[colGlat]) 
        self.R = np.sqrt(self.X**2+self.Y**2)
        self.r_gc = np.sqrt(self.R**2+self.Z**2)
        # in kpc
        self.X = self.X/1000.0
        self.Y = self.Y/1000.0
        self.Z = self.Z/1000.0
        self.R = self.R/1000.0
        self.r_gc = self.r_gc/1000.0
        self.K = dr3[colKmag]
        self.JK = dr3[colJmag]-dr3[colKmag]
        self.plateserial = dr3[colPid]
        #MK = dr3.M_K50
        #feh = dr3.feh
        return dr3
    
   # return D, Dlow, Dup, X, Y, Z, R, r_gc, K, JK, plateserial, dr3

    def nu_los(self, S1, indStar):
        '''
        functions to calculate the los density profile and all density profiles 
        calculate stellar density along a line of sight
        '''
        K1 = self.K[indStar]
        JK1 = self.JK[indStar]
        D1 = self.D[indStar]
        Dlow1 = self.Dlow[indStar] #modified the bug 20240124
        Dup1 = self.Dup[indStar] #modified the bug 20240124
        N = len(K1)
        im = np.array([np.int(i) for i in np.round((K1-self.Kgrid[0])/self.dK)])
        ic = np.array([np.int(i) for i in np.round((JK1-self.JKgrid[0])/self.dJK)])

        #print im,(im<0)
        im[im<0] = 0
        im[im>=len(self.Kgrid)] = len(self.Kgrid)-1
        ic[ic<0] = 0
        ic[ic>=len(self.JKgrid)] = len(self.JKgrid)-1
        #print ic,im
        Nm = len(self.Kgrid)
        #print np.shape(Smap)
        S = S1[im+ic*Nm]
        #print(np.shape(S))
        #print c,m
        nu = np.zeros(np.shape(self.Dgrid))
        Omega = 20./(4.*180.**2/np.pi)
        #nu_sp = np.zeros(np.shape(self.Dgrid))
        for i in range(N):
            pp1 = np.exp(-(D1[i]-self.Dgrid)**2/(2*Dlow1[i]**2))
            pp2 = np.exp(-(D1[i]-self.Dgrid)**2/(2*Dup1[i]**2))
            pp = pp1
            pp[self.Dgrid>D1[i]] = pp2[self.Dgrid>D1[i]];
            nu00= (pp/np.sum(pp)/(Omega*self.Dgrid**2))*self.dK*self.dJK
            #print(np.shape(nu00))
            nu0 = nu00*S[i]
            #print np.shape(pp),np.shape(S),np.shape(nu+nu0.reshape(np.shape(nu)))
            nu = nu+nu0#nu0.reshape(np.shape(nu));
            #nu_sp = nu_sp+nu00
        return nu

    
    def nulall(self, indPop):
        '''
        calculate stellar density for all lines of sight
        '''
        start = time.clock()
        self.nu_i = np.zeros(np.shape(self.D))
        self.indPop = indPop
        #nusp_i = np.zeros(np.shape(self.D))
        Nplate = np.shape(self.S0)
        for i in range(Nplate[0]):
            pid = self.S0[i,0]
            indStar = (self.plateserial==pid) & (self.D>0) & (self.D<self.Dgrid[-1]) & indPop
            if np.sum(indStar)>0:
                nu1 = self.nu_los(self.S0[i,1:], indStar)
                indnu = (~np.isinf(nu1)) & (nu1>0) & (~np.isnan(nu1))
                if np.sum(indnu)>2:
                    f = interp1d(self.Dgrid[indnu],nu1[indnu],bounds_error=False,fill_value=0)
                    self.nu_i[indStar] = f(self.D[indStar])
                
                print('plateID = %(i)d' % {'i':i})
        print('Time=%(t).8f' % {'t':time.clock()-start})
        #return nu_i,nusp_i

    def nulall_pltmean(self):
        '''
        calculate mean stellar density for all lines of sight and organized plate by plate
        '''
        start = time.clock()
        Nplate = np.shape(self.S0)
        nu_p = np.zeros((Nplate[0],len(self.Dgrid)))
        for i in range(Nplate[0]):
            #indP = plates.pid==i
            pid = self.S0[i,0]
            indStar = (self.plateserial==pid) & (self.D>0) & (self.D<self.Dgrid[-1])
            #print np.sum(indStar)
            if np.sum(indStar)>0:
                nu1 = self.nu_los(indStar)
                nu_p[i,:] = nu1
                print(i)
        print('Time=%(t).8f' % {'t':time.clock()-start})
        return nu_p

