# If we are building the Fortran interface, then the Fortran compile flags must
# include the openMP flag (-mp) so that local scope variables in Fortran
# routines called by the runtime are not allocated in static memory.  This makes
# the routines thread safe.  I was not able to get thread safety with -Mnosave,
# -acc=host, or -acc=multicore.
# TODO: Figure out how to generate dependence files in the build folder.
# TODO: Could not find equivalent of -fexceptions for nvfortran
CXXFLAGS_STD = -std=c++14
CUFLAGS_STD  = -std=c++14
F90FLAGS_STD = -module $(BUILDDIR) -mp
DEPFLAGS =

