# -Mrecursive needed so that we compile thread-safe Fortran routines.  This flag
# ensures that local scope variables are not allocated to static memory.
# TODO: Figure out how to generate dependence files in the build folder.
# TODO: Could not find equivalent of -fexceptions for nvfortran
CXXFLAGS_STD = -std=c++17
CUFLAGS_STD  = -std=c++17
F90FLAGS_STD = -module $(BUILDDIR) -Mrecursive
DEPFLAGS =

