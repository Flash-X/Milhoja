# TODO: Figure out how to generate dependence files in the build folder.
# TODO: Could not find equivalent of -fexceptions for nvfortran
CXXFLAGS_STD = -std=c++14
CUFLAGS_STD  = -std=c++14
F90FLAGS_STD = -module $(BUILDDIR)
DEPFLAGS =

