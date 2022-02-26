CXXFLAGS_STD = -std=c++14
CUFLAGS_STD  =
F90FLAGS_STD = -module $(BUILDDIR) -fexceptions
DEPFLAGS = -MT $@ -MMD -MP -MF $(BUILDDIR)/$*.d

