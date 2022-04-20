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
import os
import coilfitter as cf
import sparsecoil as sc
import trimesh
print(os.getcwd())
## Load existing dipole model
ccdfile = 'coildata/MagVenture_MC-B70_dipole-fit.ccd'
dpos, dmoment = cf.readccd(ccdfile)

## Identify potential dipole positions

#load stl file with coil enclosure
coilmesh = trimesh.load_mesh('coildata/MagVenture_MC-B70_dipole-fit.stl')
#mesh in mm - scale to meters
coilmesh.apply_scale(0.001)
#determine coordinate bounds (min, max) for mesh coordinate
bounds = (np.min(coilmesh.vertices, 0), np.max(coilmesh.vertices, 0))
#set dipole density
dip_density = 7/1000.
#span out cartesian grid
x,y,z = [np.arange(bounds[0][i],bounds[1][i]+dip_density,dip_density)
         for i in range(3)]
xyz = np.array(np.meshgrid(*tuple([np.arange(bounds[0][i],bounds[1][i] +
                            dip_density,dip_density) for i in range(3)]),
                           indexing='ij'))
#use trimesh to figure out if positions are inside or outside the coilmesh
inmesh=coilmesh.contains(xyz.reshape(3,-1).T)
#reshape to dipole positions by 3 (x,y,z)
coilcoords=xyz.reshape(3,-1)[:,inmesh]

#get some points in a sphere in front of the coil and a plane behind
pfront = sc.get_sph_points(r=85,d=5)[:,:3000] / 1000.
pback = sc.pos_back(coilcoords, dz=0.015)
pos = np.concatenate((pfront, pback), axis=1).T

## Get sparsified coil model
cpath = sc.sparsify(ccdfile, pos, coilcoords.T, max_n=500)

## Produce dipoles vs. reconstruction tradeoff graph via CV
# First create some additional points in front of coil for CV
pos_cv = sc.get_sph_points(r=85, d=5)[:, :5000].T / 1000.
# Placehoder for coefficients of variation
R_cv = []
# Placehoder for number of active dipoles
n = []
# Calculate A field in CV positions
dip_pos0, dip_moment0 = cf.readccd(ccdfile)
Afull = sc.A_from_dipoles(dip_moment0, dip_pos0, pos_cv)

#loop over sparsification levels
for i in range(1,cpath.shape[1],1):
    #find active dipoles
    idx = np.any(cpath[:,i].reshape(3,-1)!=0,axis=0)
    #calculate A from sparse model
    Asparse = sc.A_from_dipoles(cpath[:,i].reshape(3,-1).T[idx], coilcoords.T[idx], pos_cv)
    #coefficient of variation
    R_cv.append(1-((Asparse-Afull)**2).sum()/(Afull**2).sum())
    #active dipoles
    n.append(idx.sum())

# The following section serves to calculate the error on the surface of spheres
# at a range of distances from the coil (to approximate a head shape). It then
# calculates the fractional error for each of these surfaces with increasing
# number of dipoles. It then stops when the required error tolerance has been
# reached for all distances (max over distances). This serves to control the 
# error at the range of distances rather than overall error which would 
# effectively only consider distances very close to the coil.
# This is in accordance with what is done in the paper.

dmin = 5 #minimum distance
dmax = 84. #maximum distance
dres = 0.25 #distance resolution
tol = 1e-2 #required tolerance
#Vector of distances
dists = np.arange(dmin, dmax + dres, dres)
# Generate 500 surface point for each distance
epos = np.concatenate([sc.get_surf_points(r=85, d=d, N_gen=1000)[:,:500,None] for d in dists], axis=2)
# Calculate A field for full model at all distances
Afull = sc.A_from_dipoles(dip_moment0, dip_pos0, epos.reshape(3, -1).T)
Afull = Afull.reshape(epos.shape[1:] + (3,))
# Place holder for errors
errA = []
# Loop over sparsification levels
for i in range(1,cpath.shape[1],1):
    #find active dipoles
    idx = np.any(cpath[:,i].reshape(3,-1)!=0,axis=0)
    #calculate A from sparse model per distance
    Asparse = sc.A_from_dipoles(cpath[:, i].reshape(3, -1).T[idx], coilcoords.T[idx], epos.reshape(3, -1).T)
    Asparse = Asparse.reshape(epos.shape[1:] + (3,))
    errA.append(((Asparse - Afull) ** 2).sum((0, 2)) / (Afull ** 2).sum((0, 2)))
    if np.max(errA[-1])<tol:
        break

# Produce plot of residual variance vs. number of active dipoles
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
plt.plot(n,1-np.array(R_cv))
plt.plot(n[i-1],1-R_cv[i-1],'ro')
plt.yscale('log')
plt.xlabel('# dipoles')
plt.ylabel(r'Fraction of residual variance')

plt.subplot(2,1,2)
plt.plot(n[:i],np.max(np.array(errA),1))
plt.plot(n[i-1],np.max(errA[i-1]),'ro')
plt.plot([n[0],n[i-1]],[tol,tol],'r--')
plt.yscale('log')
plt.xlabel('# dipoles')
plt.ylabel(r'Max. fraction of residual variance')

# Extract sparsified dipole model
dip_moment_sparse = cpath[:,i].reshape(3,-1).T[idx]
dip_pos_sparse = coilcoords.T[idx]
# Save sparse model
sparseccdfile = 'coildata/MagVenture_MC-B70_dipole-fit-sparse.ccd'
cf.writeccd(sparseccdfile, dip_pos_sparse, dip_moment_sparse)

print(f'Sparsified ccd file contains {dip_pos_sparse.shape[0]} active dipoles with residual error ~{100*(1-R_cv[i-1]):.3f} % compared to non-sparse model')
