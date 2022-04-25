# dipole-fit
## Reconstruction of coil dipole models from magnetic flux density measurements
Copyright (C) 2021  Kristoffer H. Madsen

For more information about the method see:  
Electric Field Models of Transcranial Magnetic Stimulation Coils with Arbitrary Geometries: Reconstruction from Incomplete Magnetic Field Measurements  
https://arxiv.org/abs/2112.14548  
Please cite the above when using the method in publications.

Example scripts:
1) fit_MagVenture_MC-B70.py demonstrates the use of the method for reconstructing a dipole model of the MagVenture MCB-70 coil. For this purpose, flux density (B field) measurement data (in .npz format) and a mesh of the coil casing (in STL format) are provided in the coildata subfolder.
2) sparsify_MCB70_sph.py demonstrates the sparsification of a coil dipole model. For that, it loads the result of the first script (MagVenture_MC-B70_dipole-fit.ccd in the coildata subfolder) and sparsifies it to a model having the least dipoles required for achieving a given tolerance.

This package requires numpy, scipy and sklearn. The examples also require trimesh and matplotlib for processing the example data STL file and some results plotting. In addition sparsification requires the fmm3dpy package.
