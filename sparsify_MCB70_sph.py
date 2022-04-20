# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:39:45 2021

@author: khm
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
#First create some additional points in front of coil for CV
pos_cv = sc.get_sph_points(r=85, d=5)[:, :5000].T / 1000.
#placehoder for coefficients of variation
R_cv = []
#placehoder for number of active dipoles
n = []
#Calculate A field in CV positions
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

dmin = 5 #minimum distance
dmax = 84. #maximum distance
dres = 0.25 #distance resolution
tol = 1e-2 #required tolerance
#Vector of distances
dists = np.arange(dmin, dmax + dres, dres)
# Generate points of surface at various distances
epos = np.concatenate([sc.get_surf_points(r=85, d=d, N_gen=1500)[:,:1000,None] for d in dists], axis=2)
# Calculate A field for full model
Afull = sc.A_from_dipoles(dip_moment0, dip_pos0, epos.reshape(3, -1).T)
Afull = Afull.reshape(epos.shape[1:] + (3,))
# Place holder for errors
errA = []
#loop over sparsification levels
for i in range(1,cpath.shape[1],1):
    #find active dipoles
    idx = np.any(cpath[:,i].reshape(3,-1)!=0,axis=0)
    #calculate A from sparse model per distance
    Asparse = sc.A_from_dipoles(cpath[:, i].reshape(3, -1).T[idx], coilcoords.T[idx], epos.reshape(3, -1).T)
    Asparse = Asparse.reshape(epos.shape[1:] + (3,))
    errA.append(((Asparse - Afull) ** 2).sum((0, 2)) / (Afull ** 2).sum((0, 2)))

# Produce plot of residual variance vs. number of active dipoles
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.figure()
plt.plot(n,1-np.array(R_cv))
plt.yscale('log')
plt.xlabel('# dipoles')
plt.ylabel(r'Fraction of residual variance')
plt.show()
asdad
ds=(5,10,20)
epos=np.concatenate([get_sph_points_surf(d,dx=100)[...,None] for d in ds],axis=2)
Afull = ccd2nifti.A_from_dipoles(dip_moment0,dip_pos0,epos.reshape(3,-1).T)
Afull = Afull.reshape(epos.shape[1:]+(3,))
Bfull = ccd2nifti.B_from_dipoles(dip_moment0,dip_pos0,epos.reshape(3,-1).T)
Bfull = Bfull.reshape(epos.shape[1:]+(3,))
Acurve = np.zeros((len(ds),cpath.shape[1]))
Acurve[:,i] = ((Asparse-Afull)**2).sum((0,2))/(Afull**2).sum((0,2))
Bcurve = np.zeros((len(ds),cpath.shape[1]))
ndip = np.zeros((cpath.shape[1],),dtype='int')

for i in range(cpath.shape[1]):
    idx = np.any(cpath[:,i].reshape(3,-1)!=0,axis=0)
    Asparse = ccd2nifti.A_from_dipoles(cpath[:,i].reshape(3,-1).T[idx],coilcoords_s.T[idx],epos.reshape(3,-1).T)
    Asparse = Asparse.reshape(epos.shape[1:]+(3,))
    Bsparse = ccd2nifti.B_from_dipoles(cpath[:,i].reshape(3,-1).T[idx],coilcoords_s.T[idx],epos.reshape(3,-1).T)
    Bsparse = Bsparse.reshape(epos.shape[1:]+(3,))
    Acurve[:,i] = ((Asparse-Afull)**2).sum((0,2))/(Afull**2).sum((0,2))
    Bcurve[:,i] = ((Bsparse-Bfull)**2).sum((0,2))/(Bfull**2).sum((0,2))
    ndip[i]=int(idx.sum())
    print((ndip[i]))

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

plt.figure();
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#e377c2']
dtxt=[f'{ds[i]}mm distance' for i in range(len(ds))]
[plt.plot(ndip,Acurve[i],'-',color=colors[i],linewidth=2,label=dtxt[i]+' $\mathbf{A}$') for i in range(Acurve.shape[0])]
plt.yscale('log')
plt.xlabel('# active dipoles',fontsize=14)
plt.ylabel(r'residual error ($1-R^2$)',fontsize=14)

# plt.figure();
[plt.plot(ndip,Bcurve[i],'--',color=colors[i],linewidth=2,label=dtxt[i]+' $\mathbf{B}$') for i in range(Bcurve.shape[0])]
plt.yscale('log')
plt.xlabel('# active dipoles',fontsize=14)
plt.ylabel(r'residual error ($1-R^2$)',fontsize=14)
plt.legend()
plt.savefig('simulation_loop_sparse_AB.svg', dpi=600)
plt.savefig('simulation_loop_sparse_AB.png', dpi=600)

plt.figure();
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#e377c2']
dtxt=[f'{ds[i]}mm distance' for i in range(len(ds))]
[plt.plot(ndip,Acurve[i],'-',color=colors[i],linewidth=2,label=dtxt[i]) for i in range(Acurve.shape[0])]
plt.yscale('log')
plt.xlabel('# active dipoles',fontsize=14)
plt.ylabel(r'residual error ($1-R^2$)',fontsize=14)
plt.title('Residual $\mathbf{A}$',fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(1,250)
plt.legend(fontsize=12)
plt.savefig('MCB70_sparse_A.svg', dpi=600)
plt.savefig('MCB70_sparse_A.png', dpi=600)

dmin=5
dmax=84.
dres=.25
tol=1e-2
dists=np.arange(dmin,dmax+dres,dres)
epos=np.concatenate([get_sph_points_surf(d,dx=100)[...,None] for d in dists],axis=2)
Afull = ccd2nifti.A_from_dipoles(dip_moment0,dip_pos0,epos.reshape(3,-1).T)
Afull = Afull.reshape(epos.shape[1:]+(3,))
Bfull = ccd2nifti.B_from_dipoles(dip_moment0,dip_pos0,epos.reshape(3,-1).T)
Bfull = Bfull.reshape(epos.shape[1:]+(3,))
errA=[]
errB=[]
ndip=[]

for i in range(1,cpath.shape[1]):
    idx = np.any(cpath[:,i].reshape(3,-1)!=0,axis=0)
    Asparse = ccd2nifti.A_from_dipoles(cpath[:,i].reshape(3,-1).T[idx],coilcoords_s.T[idx],epos.reshape(3,-1).T)
    Asparse = Asparse.reshape(epos.shape[1:]+(3,))
    Bsparse = ccd2nifti.B_from_dipoles(cpath[:,i].reshape(3,-1).T[idx],coilcoords_s.T[idx],epos.reshape(3,-1).T)
    Bsparse = Bsparse.reshape(epos.shape[1:]+(3,))
    errA.append( ((Asparse-Afull)**2).sum((0,2))/(Afull**2).sum((0,2)))
    errB.append( ((Bsparse-Bfull)**2).sum((0,2))/(Bfull**2).sum((0,2)))
    print(f'{errA[-1].max()}, {errB[-1].max()}')
    
    if np.max([errA[-1],0*errB[-1]])<tol:
        print(f'{np.sum(idx)} dipoles sufficient with max relative error '
              f'{np.max(errA[-1])*100:.2}% and {np.max(errB[-1])*100:.2}% for A and B respectively '
              f'at distances between {dmin} and {dmax} mm.')
        break
    ndip.append(idx.sum())
dip_moment_sparse = cpath[:,i].reshape(3,-1).T[idx]
dip_pos_sparse = coilcoords_s.T[idx]
ccd2nifti.writeccd('MCB70_sparse.ccd', dip_pos_sparse, dip_moment_sparse)




plt.figure()
[plt.plot(dists,np.array(i),color='gray') for i in errA[:-1]]
plt.plot(dists,np.array(errA[-1]))
plt.plot((dmin,dmax),(tol,tol),'k--')
# plt.plot((ndip[-1],ndip[-1]),(np.min(errA),np.max(errA)),'k--')
plt.yscale('log')
plt.figure()
[plt.plot(dists,np.array(i),color='gray') for i in errB[:-1]]
plt.plot(dists,np.array(errB[-1]))
plt.yscale('log')
plt.plot((dmin,dmax),(tol,tol),'k--')
# plt.plot((ndip[-1],ndip[-1]),(np.min(errB),np.max(errB)),'k--')








'''
e=4/1000.

x,y,z=[np.arange(bounds[0][i],bounds[1][i]+e,e) for i in range(3)]
xyz=np.array(np.meshgrid(*tuple([np.arange(bounds[0][i],bounds[1][i]+e,e) for i in range(3)]),indexing='ij'))
inmesh=mesh.contains(xyz.reshape(3,-1).T)
coilcoords_s=xyz.reshape(3,-1)[:,inmesh]
tipstop = tipstop[tipstop[:,2]>0]
remdip = np.zeros(coilcoords_s.shape[1], dtype='bool')
for xi in x:
    for yi in y:
        i = np.argmin(np.sum((tipstop[:,:2]-np.array((xi,yi))[None,:])**2,1))
        #remove dipoles in z-plane within 1 mm of z tipstop
        remdip |= (coilcoords_s[0]==xi) & (coilcoords_s[1]==yi) & (coilcoords_s[2]>=(tipstop[i,2]-0.001))
        #remove dipoles in z-plane that are further away than 12mm from tipstop
        remdip |= (coilcoords_s[0]==xi) & (coilcoords_s[1]==yi) & (coilcoords_s[2]<=(tipstop[i,2]-0.012))

coilcoords_s = coilcoords_s[:,~remdip]
'''