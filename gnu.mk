CXXFLAGS_STD = -std=c++17
CUFLAGS_STD  =
F90FLAGS_STD = -J$(BUILDDIR) -fexceptions
DEPFLAGS = -MT $@ -MMD -MP -MF $(BUILDDIR)/$*.d

