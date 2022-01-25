# dipole-fit
## Reconstruction of coil dipole models from magnetic flux density measurements

For more information about the method see:  
Electric Field Models of Transcranial Magnetic Stimulation Coils with Arbitrary Geometries: Reconstruction from Incomplete Magnetic Field Measurements  
https://arxiv.org/abs/2112.14548  
Please cite the above when using the method in publications.

Running the script: fit_MCB70.py demonstrates the use of the method for reconstructing a dipole model of the MagVenture MCB-70 coil, for this purpose flux density (B field) measurement data (in .npz format) and a mesh of the coil casing (in STL format) are provided in the coildata subfolder.

This package requires numpy and scipy, and in optionally trimesh for processing the example data STL file and running the example.
