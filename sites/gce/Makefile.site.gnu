BASEDIR      = $(HOME)/Projects/OrchestrationRuntime
AMREXDIR     = $(MILHOJA_DEP_PATH)/gnu_7.5.0/AMReX_$(NDIM)D
GTESTDIR     = $(MILHOJA_DEP_PATH)/gnu_7.5.0/googletest
CUDADIR      =

CXXCOMPNAME  = gnu

CXXCOMP        = mpicxx
CXXFLAGS_PROD  = -O3 -Wuninitialized
CXXFLAGS_DEBUG = -g -O0 \
	-Wno-div-by-zero \
	-Wconversion -Wunreachable-code \
	-pedantic -Wall -Winit-self -ftree-vrp -Wfloat-equal \
	-Wunsafe-loop-optimizations -fstack-protector-all

CUCOMP       =
CUFLAGS_PROD =

F90COMP        = mpif90
F90FLAGS_PROD  = \
	-O3 -cpp -fbacktrace -march='skylake-avx512' \
    -fdefault-real-8 -fdefault-double-8 \
    -finit-real=snan -finit-derived \
    -finline-functions
F90FLAGS_DEBUG = \
	-g -O0 -cpp -fbacktrace \
    -fdefault-real-8 -fdefault-double-8 \
    -I$(BUILDDIR) \
    -finit-real=snan -finit-derived \
    -fbounds-check \
    -ffpe-trap=invalid,zero,underflow,overflow \
    -pedantic -Wall -Waliasing \
    -Wsurprising -Wconversion -Wunderflow \
    -fimplicit-none -fstack-protector-all

LDFLAGS_STD = 

CXXFLAGS_AMREX   = -I$(AMREXDIR)/include
CXXFLAGS_GTEST   = -I$(GTESTDIR)/include
CUFLAGS_AMREX    =
CUFLAGS_GTEST    =
LIB_AMREX        = -lamrex -L$(AMREXDIR)/lib
LIB_GTEST        = -lgtest -L$(GTESTDIR)/lib

LCOV = lcov
GENHTML = genhtml

CXXFLAGS_COV = -fprofile-arcs -ftest-coverage
LDFLAGS_COV  = --coverage

# Define flags for OpenMP
AMREXDIR_OMP  =
OMP_FLAGS     = -fopenmp
CU_OMP_FLAGS  =
OACC_FLAGS    =

