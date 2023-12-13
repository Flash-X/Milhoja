# -auto needed so that we compile thread-safe Fortran routines.
# We should never use -save.
# https://www.intel.com/content/www/us/en/developer/articles/technical/threading-fortran-applications-for-parallel-performance-on-multi-core-systems.html
CXXFLAGS_STD = -std=c++17
CUFLAGS_STD  =
F90FLAGS_STD = -module $(BUILDDIR) -fexceptions -auto
DEPFLAGS = -MT $@ -MMD -MP -MF $(BUILDDIR)/$*.d

