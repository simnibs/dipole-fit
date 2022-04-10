#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Copyright (C) 2021  Kristoffer H. Madsen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

import numpy as np
import time
import scipy as sp


def readccd(fn):
    """
    Basic functionality to read ccd files, ignores header

    Parameters
    ----------
    fn : string
        ccd file to parse
        
    Returns
    -------
    mpos : ndarray (n x 3)
        xyz positions of dipoles.
    m : ndarray (n x 3)
        dipole moments (xyz).
        
    On ccd file format version 1.1:
    1) First line is a header line escaped with # which can contain any number
       of variables in the form variable=value, a few of these variables are 
       reserved as can be seen below, they are separated by semicolons (;).
    2) The second line is the contains the number of dipoles expected
       (this is actually not used in practice).
    3) Third line contains a header text excaped by #, typically:
       # centers and weighted directions of the elements (magnetic dipoles)
    4-end) Remaining lines are space separated dipole positions and dipole
       moments in a string number format readable by numpy. E.g. each line must contain 
       six values: x y z mx my mz, where the first three are x,y and positions
       in meters, and the remaining three are dipole moments in x,y and z direction
       in Coulumb * meter per 1 A/s input current.
       an example could be:
       0 0 0 0 0 1.0e-03
       indicating a dipole at position 0,0,0 in z direction with strength
       0.001 C*m*s/A
    """
    dat = np.loadtxt(fn, skiprows=3)    
    mpos = dat[:,:3]
    m = dat[:,3:]
    return (mpos, m)

def writeccd(fn,mpos,m,extra=''):
    """
    Very basic functionality to write ccd files

    Parameters
    ----------
    fn : string
        filename.
    mpos : ndarray (n x 3)
        xyz positions of dipoles.
    m : ndarray (n x 3)
        dipole moments (xyz).
    extra : string, optional
        extra info to put in ccd file.

    Returns
    -------
    None.

    """
    N=m.shape[0]
    f=open(fn,'w')
    f.write('# %s version 1.1;%s\n'%(fn,extra))
    f.write('%i\n'%N)
    f.write('# centers and weighted directions of the elements (magnetic dipoles)\n')
    for i in range(N):
        f.write('%.15e %.15e %.15e '%tuple(mpos[i]))
        f.write('%.15e %.15e %.15e\n'%tuple(m[i]))
    f.close()
        

def ABdip(points,Dpos,calcA=True,calcB=True,verbose=False):
    """
    Leadfield A and B field for dipoles

    Parameters
    ----------
    points : ndarray
        DESCRIPTION.
    Dpos : ndarray
        DESCRIPTION.
    calcA : Boolean, optional
        Return A field. The default is True.
    calcB : Boolean, optional
        Return B field. The default is True.
    verbose : TYPE, optional
        Show additional information on time. The default is False.

    Returns
    -------
    ndarrays containing A and/or B lead fields depending on calcA and calcB

    """
    tt=time.time()
    points=np.asarray(points)
    Dpos=np.asarray(Dpos)

    if calcB:
        B=np.zeros((3,points.shape[0],3,Dpos.shape[0]),dtype=np.float64,order='C')
    if calcA:
       A=np.zeros((3,points.shape[0],3,Dpos.shape[0]),dtype=np.float64,order='C')
    if not(calcA) and not(calcB): return None    
    rv=points[:,None,:]-Dpos[None,...]
    r=((rv**2).sum(2)**0.5)[...,None]
    ir=1.0e-7*r**(-3)
    re=rv/r
    if calcB:
        B[0,:,0,:] =              (3.*(re[...,0]*re[...,0])-1.)*ir[...,0]
        B[0,:,1,:] = B[1,:,0,:] = (3.*(re[...,1]*re[...,0]))   *ir[...,0]
        B[0,:,2,:] = B[2,:,0,:] = (3.*(re[...,2]*re[...,0]))   *ir[...,0]
        
        B[1,:,1,:] =              (3.*(re[...,1]*re[...,1])-1.)*ir[...,0]
        B[1,:,2,:] = B[2,:,1,:] = (3.*(re[...,2]*re[...,1]))   *ir[...,0]
        
        B[2,:,2,:] =              (3.*(re[...,2]*re[...,2])-1.)*ir[...,0]
    if calcA:
        A[0,:,0,:] = 0. 
        A[0,:,1,:] =  re[...,2]*ir[...,0]*r[...,0]
        A[0,:,2,:] = -re[...,1]*ir[...,0]*r[...,0]
        
        A[1,:,0,:] = -A[0,:,1,:]         
        A[1,:,1,:] = 0.
        A[1,:,2,:] =  re[...,0]*ir[...,0]*r[...,0]
        
        A[2,:,0,:] = -A[0,:,2,:]        
        A[2,:,1,:] = -A[1,:,2,:]        
        A[2,:,2,:] = 0.
    if verbose:
        print('time passed: %.2fs'%(time.time()-tt))
    
    if calcB:
        if calcA: return (A,B)
        else: return B
    else: return A

def ABdip_z(points,Dpos,calcA=True,calcB=True,verbose=False):
    """
    Leadfield A and B field for dipoles z direction only

    Parameters
    ----------
    points : ndarray
        DESCRIPTION.
    Dpos : ndarray
        DESCRIPTION.
    calcA : Boolean, optional
        Return A field. The default is True.
    calcB : Boolean, optional
        Return B field. The default is True.
    verbose : TYPE, optional
        Show additional information on time. The default is False.

    Returns
    -------
    ndarrays cotaining A and/or B lead fields depending on calcA and calcB

    """
    tt=time.time()
    points=np.asarray(points)
    Dpos=np.asarray(Dpos)
    if calcB:
        B=np.zeros((3,points.shape[0],Dpos.shape[0]),dtype=np.float64,order='C')
    if calcA:
        A=np.zeros((3,points.shape[0],Dpos.shape[0]),dtype=np.float64,order='C')
    if not(calcA) and not(calcB): return None    
    rv=points[:,None,:]-Dpos[None,...]
    r=((rv**2).sum(2)**0.5)[...,None]
    ir=1.0e-7*r**(-3)
    re=rv/r
    if calcB:
        B[0,:,:] = (3.*(re[...,2]*re[...,0]))   *ir[...,0]
        B[1,:,:] = (3.*(re[...,2]*re[...,1]))   *ir[...,0]
        B[2,:,:] = (3.*(re[...,2]*re[...,2])-1.)*ir[...,0]
    if calcA:
        A[0,:,:] = -re[...,1]*ir[...,0]*r[...,0]        
        A[1,:,:] =  re[...,0]*ir[...,0]*r[...,0]
        A[2,:,:] = 0. 
    if verbose:
        print('time passed: %.2fs'%(time.time()-tt))
    if calcB:
        if calcA: return (A,B)
        else: return B
    else: return A    
    
def ABdip_z_dir(points, pdir, Dpos, verbose=False):
    tt=time.time()
    points=np.asarray(points)
    Dpos=np.asarray(Dpos)
    B=np.zeros((points.shape[0],Dpos.shape[0]),dtype=np.float64,order='C')

    rv=points[:,None,:]-Dpos[None,...]
    r=((rv**2).sum(2)**0.5)[...,None]
    ir=1.0e-7*r**(-3)
    re=rv/r
    B[pdir==0] = (3.*(re[...,2]*re[...,0]))   *ir[...,0]
    B[pdir==1] = (3.*(re[...,2]*re[...,1]))   *ir[...,0]
    B[pdir==3] = (3.*(re[...,2]*re[...,2])-1.)*ir[...,0]
    
    if verbose:
        print('time passed: %.2fs'%(time.time()-tt))
    return B

def interp_z(B,pos,coilcoords):
    """
    Utility to interpolate B field per plane in an cartesian equidistant 
    in-plane grid for visualization of residual

    Parameters
    ----------
    B : ndarray
        B field.
    pos : ndarray
        positions of B field.
    coilcoords : ndarray
        coordiates of dipoles.

    Returns
    -------
    Bi : ndarray
        Interpolated B field.

    """
    import scipy.interpolate as sint
    x = np.unique(pos[:,0])
    xi = np.arange(x[0],x[-1]+np.diff(x).min(),0.003)
    y = np.unique(pos[:,1])
    yi = np.arange(y[0],y[-1]+np.diff(y).min(),0.003)
    xyi = np.array(np.meshgrid(xi,yi,indexing='ij'))
    zi = np.unique(pos[:,2])
    Bi = np.empty((len(xi), len(yi), len(zi), B.shape[1]), dtype=np.float64())
    for i, z in enumerate(np.unique(pos[:,2])):
        for k in range(Bi.shape[-1]):
            Bi[...,i,k] = sint.griddata(pos[pos[:,2] == z][:,:2],B[pos[:,2] == z, k],
                                        xyi.reshape(2,-1).T, fill_value=np.nan,
                                        method='nearest').reshape(xyi[0].shape)
        if z<=np.max(coilcoords[2]) and z >= np.min(coilcoords[2]):
            distc = np.min(np.abs(xi[:,None] - coilcoords[0,None])[:,None,:] + \
            np.abs(yi[:,None]-coilcoords[1,None])[None,:,:] + \
            np.abs(z - coilcoords[2]),-1)
            Bi[distc<0.012,i,:] = np.nan
    return Bi

def fit_coil_Nsplit(pos, coilcoords, Bs, n=2, flat=True, plotfn=None,alphas=np.logspace(-5,2,100)):
    """
    Main function to fit dipole model

    Parameters
    ----------
    pos : ndarray (n x 3)
        positions of measurements in meters.
    coilcoords : ndarray (ndip x 3)
        positions of dipoles.
    Bs : list of ndarrays (each n x 3)
        Instances of the B fiels at positions (pos) measured in Teslas, typically this just be a list of one element, 
        but if there are several repeats or it is a simulation this can be more than one element, 
        this basically speeds the process up as only one inversion is needed for multiple datasets.
    n : integer, optional
        Number of CV splits. The default is 2.
    flat : Boolean, optional
        if this is set to True, only fit dipoles in z direction, otherwise estimate all 3 cartesian parameters. The default is True.
    plotfn : string, optional
        Filename to output plots in (pdf format). The default is None.
    alphas : ndarray, optional
        l2 regularization parameters to optimize over. The default is np.logspace(-5,2,100).
(solution,(alphas, testerrv,  mcorrv, propt))
    Returns
    -------
    solution: ndarray (ndip)
        The estimated dipole moments for each B field input
    alphas: ndarray
        Vector of alphas, for plotting
    testerrv: ndarray
        Estimated test error (P) for each alpha, for plotting
    mcorrv: ndarray
        Estimated reproducbility (R) for each alphas, for plotting
    propt: list
        index of alpha for best PR tradeoff for each B field input, for plotting

    """
    #just grab the first B
    Bin = Bs[0]
    #number of B field measurements (each 3 dimensions)
    Mn = int(Bin.shape[0])
    #fraction for splitting
    SplitFrac = 1. / n
    #indices for permuting measurements
    idx = np.random.permutation(np.arange(Mn))
    #get random indices for splitting
    idx_splits = [idx[int(i * SplitFrac * Mn):int(SplitFrac * (i + 1) * Mn)] for i in range(n)]
    
    #K, number of dipoles
    if flat:
        K = int(coilcoords.shape[1])
    else:
        K = 3 * int(coilcoords.shape[1])
    #number of B field measurements (typically just one)
    M = len(Bs)
    #initialize solutions and place holder for unexplained variance
    sols = np.zeros((M,n, len(alphas),K),dtype=np.float64)
    #unexpl variance for each B, a splits by splits matrix for each alpha
    uexpl_var = np.zeros((M,n, n, len(alphas)))
    #loop over splits
    for i in range(n):
        print('Split %i of %i' % (i+1,n))
        #get B leadfield
        if flat:
            XB = ABdip_z(pos[idx_splits[i]], coilcoords.T, calcB=True, calcA=False)
        else:
            XB = ABdip(pos[idx_splits[i]], coilcoords.T, calcB=True, calcA=False)
        #N, no of measurements for split (B field 3 times B field positions)
        N = 3 * len(idx_splits[i])
        #unfold across dimensions to create design matrix for linear problem
        XB = XB.reshape(N,K)
        t0=time.time()
        # perform SVD for design matrix (allows fast solution for all alphas)
        U, s, Vt = sp.linalg.svd(XB,full_matrices=False)
        print('SVD time: %.1fs'%(time.time()-t0))
        #vectorize actual measurements for split
        Bt = np.array([Bs[m][idx_splits[i]].T.reshape(N) for m in range(M)]).T
        UtB = U.T@Bt
        #function to solve for each aplha
        l2solve = lambda a:((s / (s ** 2 + a))*UtB.T)@Vt
        #solve for each alpha
        for j, a in enumerate(alphas):
            sols[:,i,j] = l2solve(a)
        BtB = np.sum(Bt**2,axis=0)
        #function to calculate fraction of sum of sqaures error
        def err(J):
            return np.sum(((J@XB.T).reshape(Bt.shape[1],-1,Bt.shape[0])-
                           Bt.T[:,None,:])**2,axis=-1)/BtB[:,None]
        #calculate explained variance for each alpha for the previous solutions
        #for first split this would basically just be the unexplained variance within split
        #this within split variance is not actually used for anything, but next split it would be
        #both the one within split and between split 1 and 2, in effect this fills the lower 
        #triangular part of of the split by split unexpected variance matrix for each alpha
        for j, sol in enumerate(sols[0,:i+1]):
            uexpl_var[:,i,j,:] = err(sols[:,j].reshape(-1,sols.shape[-1]))
    #this fills the upper triangular part of the above, this is basically done in this way
    #to lower memory requirements. The old leadfield is formed again and the identified solution is used.    
    for i in range(n-1):
        print('Calculating test error pairs %i of %i' % (i+1,n-1))
        if flat:
            XB = ABdip_z(pos[idx_splits[i]], coilcoords.T, calcB=True, calcA=False)
        else:
            XB = ABdip(pos[idx_splits[i]], coilcoords.T, calcB=True, calcA=False)
        N = 3 * len(idx_splits[i])
        XB = XB.reshape((N,K))
        Bt = np.array([Bs[m][idx_splits[i]].T.reshape(N) for m in range(M)]).T
        BtB = np.sum(Bt**2,axis=0)
        def err(J):
            return np.sum(((J@XB.T).reshape(Bt.shape[1],-1,Bt.shape[0])-
                           Bt.T[:,None,:])**2,axis=-1)/BtB[:,None]
        for j, sol in enumerate(sols[0,i+1:]):
            uexpl_var[:,i,j+i+1,:] = err(sols[:,i+1+j].reshape(-1,sols.shape[-1]))
    #Calculate relative mean squared error
    expl_MSE = 1 - (uexpl_var)
    #initialize variable for MSE/correlation between solutions (reliability)
    corrA=np.zeros((M,len(alphas),n,n))
  
    solution=[]
    testerrv=[]
    mcorrv=[]
    propt=[]
    #calculate all pairwise reliabilities, in principle there is no reason to calculate the diagonal,
    #which is 1 and also only the upper triangular part is needed as it is symmetric.
    for m in range(M):
        for j, a in enumerate(alphas):
            for i in range(n):
                 for ii in range(n):
                     sol1 = sols[m][i][j]
                     sol2 = sols[m][ii][j]
                     #one minus sum of sqaures (SS) difference of the two solutions divided by the SS mean solution.
                     corrA[m][j][ii, i] = 1 - ((((sol1 - sol2)**2).sum()) / (((0.5 * sol1 + 0.5 * sol2)**2).sum()))
        #mean of upper triangular part (between split reliability), for two splits this is just one number
        mcorr = np.array([np.mean(corrA[m][j][np.triu_indices(n,1)]) for j, a in enumerate(alphas)])
        #mean of off diagnonal elements of testerror (predictability), this is assymetric in general.
        #this is between split test error
        testerr = np.array([0.5*np.mean(expl_MSE[m][:,:,j][np.triu_indices(n,1)]) + 
                   0.5*np.mean(expl_MSE[m][:,:,j][np.tril_indices(n,1)])
                   for j, a in enumerate(alphas)])
        #same but for unexplained variance, here just used for outputting
        testvar = np.array([0.5*np.mean((1-uexpl_var[m])[:,:,j][np.triu_indices(n,1)]) + 
                   0.5*np.mean((1-uexpl_var[m])[:,:,j][np.tril_indices(n,1)])
                   for j, a in enumerate(alphas)])
        #index of model with best predictability (P)
        pidx = np.argmax(testerr)
        #index of model with best predictability/reliability tradeoff (PR)
        pridx = np.argmax(testerr +  mcorr)
        propt.append(pridx)
        testerrv.append(testerr)
        mcorrv.append(mcorr)
        print('Residual unexplained variance (Best/PR tradeoff): %.2f%%, %.2f%%' % 
              ((1 - testvar[pidx]) * 100., (1 - testvar[pridx]) * 100.))
        #the alpha value for obtaining best tradeoff
        pralpha = alphas[pridx]
        
        #produce PR plot if requested
        if not plotfn is None:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            pdf = PdfPages(plotfn + '_report.pdf')
            fig = plt.figure()
            plt.plot(mcorr,testerr,'-o',markersize=3)
            plt.ylabel(r'Predictability ($R^2$)')
            plt.xlabel(r'Reliability ($R^2$)')

            plt.plot(mcorr[pridx],testerr[pridx],'ro')
            plt.plot(mcorr[pidx],testerr[pidx],'gx')
            lastpos = (0,0)
            rrange = mcorr.max() - mcorr.min()
            prange = testerr.max() - testerr.min()
            for j, a in enumerate(alphas):
                if mcorr[j]>0 and (np.abs(mcorr[j] - lastpos[0]) / rrange +\
                   np.abs(testerr[j] - lastpos[1]) / prange) >= 0.25:
                       plt.plot(mcorr[j], testerr[j],'k.')
                       plt.text(mcorr[j] + 0.002 * rrange + 0.005 * rrange, testerr[j],r'$\alpha=10^{%.1f}$'%np.log10(a))
                       lastpos = (mcorr[j], testerr[j])
            plt.title('Prediction/Reliability curve')
            plt.xlim(mcorr.min()-rrange/20,1+rrange/20)
            plt.ylim(testerr.min()-prange/20,1+prange/15)

            plt.plot((0.98,1),(1,1),'k--')
            plt.plot((1,1),(0.98,1),'k--')
            plt.xlim(0,1.1)
            plt.ylim(0,1.1)
            plt.tight_layout()
            pdf.savefig(fig)
            fig = plt.figure()
            plt.plot(alphas, -np.log10(1-testerr))
            plt.plot(pralpha, -np.log10(1-testerr[pridx]), 'ro')
            plt.text(pralpha, -np.log10(1-testerr[pridx]), 'Predictability/Reliability tradeoff')
            plt.plot(alphas[pidx], -np.log10(1-testerr[pidx]), 'ko')
            plt.text(alphas[pidx], -np.log10(1-testerr[pidx]), 'Predictability optimum')
            plt.xscale('log')
            plt.xlabel(r'$\alpha$')
            plt.ylabel(r'$-$log$_{10}$(expl. MSE)')
    
            pdf.savefig(fig)
            pdf.close()
        #calculate mean solution across splits, which will be the main output
        solution.append(np.sum([sols[m][i][pridx] / n for i in range(n)], 0))
    del sols
    return (solution,(alphas, testerrv,  mcorrv, propt))

    
if __name__ == '__main__':
    pass
    