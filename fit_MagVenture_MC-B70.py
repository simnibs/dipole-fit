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
import trimesh
import coilfitter as cf
# load stl file with coil enclosure
coilmesh = trimesh.load_mesh('coildata/MagVenture_MC-B70_dipole-fit.stl')
# mesh in mm - scale to meters
coilmesh.apply_scale(0.001)

# determine coordinate bounds (min, max) for mesh coordinate
bounds = (np.min(coilmesh.vertices, 0), np.max(coilmesh.vertices, 0))
# set dipole density (5 mm was used in the paper, but requires more memory and time)
dip_density = 7/1000.

# span out cartesian grid
x, y, z = [np.arange(bounds[0][i], bounds[1][i]+dip_density, dip_density)
           for i in range(3)]
xyz = np.array(np.meshgrid(*tuple([np.arange(bounds[0][i], bounds[1][i] +
                           dip_density, dip_density) for i in range(3)]),
                           indexing='ij'))

# use trimesh to figure out if positions are inside or outside the coilmesh
inmesh = coilmesh.contains(xyz.reshape(3, -1).T)
# reshape to dipole positions by 3 (x,y,z)
coilcoords = xyz.reshape(3, -1)[:, inmesh]

# load measurement data, contains B field measurement i x,y and z direction (data['B'] a N x 3 array)
# and dipole (x,y,z) positions in mm (N x 3 array)
data = np.load('coildata/MagVenture_MC-B70_dipole-fit.npz')

# scale positions to meter and cast to 64-bit float for precision
pos = data['xyz'].astype(np.float64)/1000.
B = data['data'].astype(np.float64)

# how many B field measurement points should be used - using all (~100k) is
# recommended but will use quite a lot of memory (~16GB), subsampling with a
# factor of 3 will use 32k measurement points and require ~5GB
subsample_factor = 3
B = B[::subsample_factor]
pos = pos[::subsample_factor]

# select only dipoles further away than 6 mm from any measurement as field approximation is known to fail near dipoles
# use list comprehension to save memory instead of broadcasting
cidx = np.sqrt([np.min(((pos-cpos[None])**2).sum(1)) for cpos in coilcoords.T]) > (6/1000.)
coilcoords = coilcoords[:, cidx]

# note that this setup will not produce identical results to the ones in the paper, because
# there is a further restriction that dipoles do not cover the entire casing there, but rather just
# the upper part (CT scan shows only wires there), this further restriction does not really influence the fit
# but creates a more compact coilmodel and reduces memory requirements. Note that fitting all of the (100k)
# measurements requires a lot of memory, this can be reduced by decreasing the number of measurements or
# dipoles positions if desired. Increasing the number of splits will also reduce memory footprint.

<<<<<<< HEAD
#run coilfitting with two splits with free dipole directions, note that B field is a list of one element
#expected runtime on modern computer (year 2020) a few minutes.
solutions,out_real=cf.fit_coil_Nsplit(pos, coilcoords, [B], n=2, flat=False, plotfn=None)
#reshape and transpose to form ndip x 3 array
dip_moments = solutions[0].reshape(3, -1).T

#write ccd file to disk
=======
# run coilfitting with two splits with free dipole directions, note that B field is a list of one element
# expected runtime on modern computer (year 2020) a few minutes.
solutions, out_real = cf.fit_coil_Nsplit(pos, coilcoords, [B], n=2, flat=False, plotfn=None)
# reshape and transpose to form ndip x 3 array
dip_moments = solutions[0].reshape(3, -1).T

# write ccd file to disk
>>>>>>> refs/remotes/origin/main
ccdfile = 'coildata/MagVenture_MC-B70_dipole-fit.ccd'
cf.writeccd(ccdfile, coilcoords.T, dip_moments)
