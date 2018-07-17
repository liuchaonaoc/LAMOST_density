# LAMOST_density
=== === ===

Updated code since 2018-07-17

--- --- ---

LM_density_DR5.py: New density estimation for DR5 (can also be used for DR3/DR4)

LMdensity_DR5_demo.py: A demonstration about how to use the python code to derive the stellar density. It needs two data files: 

    DR3_KGiants_short_Wang18.fits: the K giant stras used in Wang et al. 2018, MNRAS, 478, 3367.
    
    Selection_plates_DR5.csv: contains the selection function S^(-1) for all DR5 plates (including most of the DR3 plates)

To estimate stellar density for each plate, you are required to use the serial number of each plate rather than the name of the plate. Normally, the published LAMOST catalog only contains "planID" as the name of the plate. To find the corresponding serial number of the plate, you can match any LAMOST data catalog with the Observed Plate Information Catalog using "planID" for each object in your own data catalog. Then use the field "pID" (I rename it as "plateserial" in the demo data file) in the Plate Information catalog as the serial number of plates. 

The Observed Plate Information Catalog can be found at dr3.lamost.org/catalogue.

=== === ===

Out of date

--- --- ---

Measure stellar density profile for te LAMOST selected stellar populations

Selection_plates.csv contains the S^(-1) for all DR3 plates

LM_density.py contains the functions to calculate the stellar density

haloRGB.py: a sample code to demonstrate how to calculate the density for the halo-like RGB stars (Xu et al.)

LMDR3_haloRGB2.dat: the final catalog of the halo-like RGB stars, same as Table 4 in Liu et al.

diskRGB.py: a sample code to demonstrate how to calculate the density for the disk-like RGB stars (Wang et al.)

LMDR3_diskRGB2.dat: the final catalog of the disk-like RGB stars, same as Table 3 in Liu et al.

