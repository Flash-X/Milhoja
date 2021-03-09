#####----- Integral quantities
IQ_TIME_IDX = 0
IQ_MASS_IDX = 1
IQ_XMOM_IDX = 2
IQ_YMOM_IDX = 3
IQ_ZMOM_IDX = 4
IQ_ETOT_IDX = 5
IQ_E_KE_IDX = 6
IQ_EINT_IDX = 7

# The X, Y, Z momentum should be zero within round-off noise
IQ_THRESHOLD_VEL  = 2.5e-16
IQ_THRESHOLD_MASS = 2.5e-15
IQ_THRESHOLD_ETOT = 2.5e-12

# The non-conserved integrated energies should match reasonably well across runs
IQ_THRESHOLD_KE   = 7.5e-16
IQ_THRESHOLD_EINT = 1.0e-12

#####----- Working with YT
YT_XAXIS = 0
YT_YAXIS = 1
YT_ZAXIS = 2

# The variables to study and visualize
# The integer is the value to use for indexing datasets as var000X
# Mapping of variable to var000X can be found in writeMultiPlotfile
# in Grid_AmrCoreFlash.cpp
YT_VAR_LUT = [(1,  'Density'), \
              (2,  'Internal E'), \
              (3,  'Energy'), \
              (6,  'Pressure'), \
              (7,  'Temperature'), \
              (8,  'X-Velocity'), \
              (9,  'Y-Velocity'), \
              (10, 'Z-Velocity')]
YT_N_VARS = len(YT_VAR_LUT)

