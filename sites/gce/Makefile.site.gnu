BASEDIR      = $(HOME)/Projects/OrchestrationRuntime
AMREXDIR     = $(MILHOJA_DEP_PATH)/gnu_7.5.0/AMReX_$(NDIM)D
GTESTDIR     = $(MILHOJA_DEP_PATH)/gnu_7.5.0/googletest
CUDADIR      =

CXXCOMPNAME  = gnu
CXXCOMP      = mpicxx

CXXFLAGS_PROD    = -O3
CXXFLAGS_DEBUG   = -g -O0 -DGRID_LOG

CUCOMP       =
CUFLAGS_PROD =

LDFLAGS_STD = -lpthread -lstdc++ -lgfortran

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

