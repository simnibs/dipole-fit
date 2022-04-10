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
import coilfitter as cf
import sklearn.linear_model as lin
import fmm3dpy

def A_from_dipoles(d_moment, d_position, target_positions, eps=1e-3, direct='auto'):
    '''
    Get A field from dipoles using FMM3D

    Parameters
    ----------
    d_moment : ndarray
        dipole moments (Nx3).
    d_position : ndarray
        dipole positions (Nx3).
    target_positions : ndarray
        position for which to calculate the A field.
    eps : float
        Precision. The default is 1e-3
    direct : bool
        Set to true to force using direct (naive) approach or False to force use of FMM.
        If set to auto direct method is used for less than 300 dipoles which appears to be faster i these cases.
        The default is 'auto'

    Returns
    -------
    A : ndarray
        A field at points (M x 3) in Tesla*meter.

    '''
    #if set to auto use direct methods if # dipoles less than 300
    if direct=='auto':
        if d_moment.shape[0]<300:
            direct = True
        else:
            direct = False
    if direct is True:
        out = fmm3dpy.l3ddir(charges=d_moment.T, sources=d_position.T,
                  targets=target_positions.T, nd=3, pgt=2)
    elif direct is False:
        #use fmm3dpy to calculate expansion fast
        out = fmm3dpy.lfmm3d(charges=d_moment.T, eps=eps, sources=d_position.T,
                  targets=target_positions.T, nd=3, pgt=2)
    else:
        print('Error: direct flag needs to be either "auto", True or False')
    A = np.empty((target_positions.shape[0], 3), dtype=float)
    #calculate curl
    A[:, 0] = (out.gradtarg[1][2] - out.gradtarg[2][1])
    A[:, 1] = (out.gradtarg[2][0] - out.gradtarg[0][2])
    A[:, 2] = (out.gradtarg[0][1] - out.gradtarg[1][0])
    #scale
    A *= -1e-7
    return A

def B_from_dipoles(d_moment, d_position, target_positions, eps=1e-3, direct='auto'):
    '''
    Get B field from dipoles using FMM3D

    Parameters
    ----------
    d_moment : ndarray
        dipole moments (Nx3).
    d_position : ndarray
        dipole positions (Nx3).
    target_positions : ndarray
        positions for which to calculate the B field.
    eps : float
        Precision. The default is 1e-3
    direct : bool
        Set to true to force using direct (naive) approach or False to force use of FMM.
        If set to auto direct method is used for less than 300 dipoles which appears to be faster i these cases.
        The default is 'auto'

    Returns
    -------
    B : ndarray
        B field at points (M x 3) in Tesla.

    '''
    #if set to auto use direct methods if # dipoles less than 300
    if direct=='auto':
        if d_moment.shape[0]<300:
            direct = True
        else:
            direct = False
    if direct is True:
        out = fmm3dpy.l3ddir(dipvec=d_moment.T, sources=d_position.T,
                  targets=target_positions.T, nd=1, pgt=2)
    elif direct is False:
        out = fmm3dpy.lfmm3d(dipvec=d_moment.T, eps=eps, sources=d_position.T,
                  targets=target_positions.T, nd=1, pgt=2)
    else:
        print('Error: direct flag needs to be either "auto", True or False')
    B = out.gradtarg.T
    B *= -1e-7
    return B

def make_design(pos, dip_positions, calcB=True, calcA=False):
    """
    Create A and B leadfields for dipole positions
    
    Parameters
    ----------
    pos : ndarray (N x 3)
        positions of A,B field.
    dip_positions : ndarray
        coordiates of dipoles (M x 3).
    calcB : bool
        return B leadfield
    calcA : bool
        return A leadfield
    Returns
    -------
    XA : ndarray
        Leadfield matrix (3N x 3M).

    """
    if calcA and calcB:
        XA,XB = cf.ABdip(pos, dip_positions, calcA=True, calcB=True)
        XA = XA.reshape(3 * int(pos.shape[0]), 3 * int(dip_positions.shape[0]))
        XB = XB.reshape(3 * int(pos.shape[0]), 3 * int(dip_positions.shape[0]))
        return XA, XB
    elif calcA:
        XA = ABdip(pos, dip_positions, calcA=True, calcB=False)
        XA = XA.reshape(3 * int(pos.shape[0]), 3 * int(dip_positions.shape[0]))
        return XA
    elif calcB:
        XB = ABdip(pos, dip_positions, calcA=False, calcB=True)
        XB = XB.reshape(3 * int(pos.shape[0]), 3 * int(dip_positions.shape[0]))
        return XB

def get_sph_points(r=85, d=10, N_gen=20000):
    """
    Basic and very inefficient way to return uniform points in the negative half sphere
    with the center at (0,0,r) and radius r-d
    
    Parameters
    ----------
    r : float
        sphere center z position.
    d : float
        min distance from surface.
    N_gen : int
    Number of points to generate, note that the return number of points will be smaller - 
    approximately resulting in N = N_gen*pi/6 points
    
    Returns
    -------
    epos : ndarray
        Positions.
    """
    #Uniform points
    epos = (np.random.rand(3, N_gen) - 0.5) * 2
    #Reject point outside unit sphere and flip to negative half
    idx = (np.sum(epos**2,0)<=1)
    epos = epos[:,idx] * np.sign((epos[2,idx]<0) - 0.5)[None]
    #Scale to r
    epos *= (r-d)
    #move center
    sph_center = np.array((0, 0, r))
    epos += sph_center[:,None]
    return epos

def pos_back(coilcoords,dz=0.015, N=300):
    """
    Return a plane of random points with the size of the coil behind the coil at distance -dz
    """
    
    zback = coilcoords.min(1)[2] - dz
    posmin = coilcoords.min(1)
    posmax = coilcoords.max(1)
    posrange = posmax-  posmin
    posback = (np.random.rand(2,N) - 0.5) * posrange[:2,None] + posmin[:2,None]
    posback = np.concatenate((posback,np.ones((1,posback.shape[1]))*zback), axis=0)
    return posback

        
def sparsify(ccdfile, pos, dip_positions, max_n=500):
    """
    Sparsify coilmodel with full path via Orthogonal Matching Pursuit
    Parameters
    ----------
    ccdfile : string (filename)
        location of ccdfile.
    pos : ndarray
        Positions where to approximate A and B fields.
    max_n : int
        Maximum number of active dipole coefficients.
    Returns
    ----------
    coefs : ndarray
        Regularization path of dipole coefficients
    """
    dip_pos0, dip_moment0 = cf.readccd(ccdfile)
    yA = A_from_dipoles(dip_moment0, dip_pos0, pos).ravel(order='F')
    yB = B_from_dipoles(dip_moment0, dip_pos0, pos).ravel(order='F')
    XA, XB = make_design(pos, dip_positions, calcA=True, calcB=True)
    scaleA = np.linalg.norm(yA)
    scaleB = np.linalg.norm(yB)
    X = np.concatenate((XA/scaleA, XB/scaleB),axis=0)
    del XA
    del XB
    y = np.concatenate((yA/scaleA, yB/scaleB), axis=0)
    del yA
    del yB
    scaleX = np.linalg.norm(X)
    scaley = np.linalg.norm(y)
    coefs = lin.orthogonal_mp(X/scaleX, y/scaley, 
                      n_nonzero_coefs=max_n, return_path=True)
    coefs *= scaley / scaleX
    return coefs