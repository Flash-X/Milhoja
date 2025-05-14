# It is intended that make be run from the same folder as the Makefile.  All
# executions of make that build the library should start from a clean build
# since this Makefile does not yet build up and account for dependencies.
#TODO: Add in full use of dependency files carefully.
#TODO: Make Makefile.configure here?
#TODO: Let users specify if they want to include the Fortran interface.
#TODO: At the moment, this script will overwrite whatever flag specifications
#      that a user might give at the command line (e.g., CXXFLAGS).  Is this
#      acceptable?  If not, how to handle this?  Perhaps it's best to put this
#      off until it is understood how this could be integrated with spack.

SHELL=/bin/sh

.SUFFIXES:

INCDIR          := ./includes
SRCDIR          := ./src
BUILDDIR        := ./build
INTERFACEDIR    := ./interfaces
TOOLSDIR        := ./tools
CONFIG_MAKEFILE := ./Makefile.configure

include $(CONFIG_MAKEFILE)

BACKEND_BUILDDIR :=
ifeq ($(RUNTIME_BACKEND),CUDA)
BACKEND_BUILDDIR := $(BUILDDIR)/CudaBackend
else ifeq ($(RUNTIME_BACKEND),HOSTMEM)
BACKEND_BUILDDIR := $(BUILDDIR)/FakeCudaBackend
else ifeq ($(RUNTIME_BACKEND),OMPTARGET)
BACKEND_BUILDDIR := $(BUILDDIR)/OmpTargetBackend
endif

GRID_BACKEND := $(shell echo $(GRID_BACKEND) | tr A-Z a-z)

LIBNAME ?= milhoja
TARGET          := $(BUILDDIR)/lib$(LIBNAME).a

MILHOJA_H       := $(BUILDDIR)/Milhoja.h

include $(SITE_MAKEFILE)

MAKEFILES := ./Makefile $(CONFIG_MAKEFILE) $(SITE_MAKEFILE)

# Default shell commands
RM ?= /bin/rm

ifeq      ($(CXXCOMPNAME),gnu)
include ./gnu.mk
else ifeq ($(CXXCOMPNAME),intel)
include ./intel.mk
else ifeq ($(CXXCOMPNAME),nvhpc)
include ./nvhpc.mk
else
$(error $(CXXCOMPNAME) compiler not yet supported.)
endif

SIZES_JSON      := $(if $(SUPPORT_PACKETS),$(BUILDDIR)/sizes.json)

ifeq ($(DEBUG),true)
CXXFLAGS = -I$(INCDIR) -I$(BUILDDIR) $(CXXFLAGS_STD) $(CXXFLAGS_DEBUG) $(CXXFLAGS_AMREX)
F90FLAGS = -I$(BUILDDIR) $(F90FLAGS_STD) $(F90FLAGS_DEBUG)
else
CXXFLAGS = -I$(INCDIR) -I$(BUILDDIR) $(CXXFLAGS_STD) $(CXXFLAGS_PROD)  $(CXXFLAGS_AMREX)
F90FLAGS = -I$(BUILDDIR) $(F90FLAGS_STD) $(F90FLAGS_PROD)
endif

CPP_SRCS := $(wildcard $(SRCDIR)/Milhoja_*.cpp)
CPP_HDRS := $(wildcard $(INCDIR)/Milhoja_*.h)
CPP_SRCS_GRID := $(wildcard $(SRCDIR)/Milhoja_Grid*.cpp $(SRCDIR)/Milhoja_*Iter*.cpp)
CPP_HDRS_GRID := $(wildcard $(INCDIR)/Milhoja_Grid*.h $(INCDIR)/Milhoja_*Iter*.h)

ifeq ($(SUPPORT_PUSH),)
CPP_SRCS := $(filter-out $(SRCDIR)/Milhoja_TileFlashxr.cpp,$(CPP_SRCS))
CPP_HDRS := $(filter-out $(INCDIR)/Milhoja_TileFlashxr.h $(INCDIR)/Milhoja_FlashxrTileRaw.h,$(CPP_HDRS))
endif
ifeq ($(SUPPORT_EXEC),)
CPP_SRCS := $(filter-out $(SRCDIR)/Milhoja_TileAmrex.cpp,$(CPP_SRCS))
CPP_HDRS := $(filter-out $(INCDIR)/Milhoja_TileAmrex.h,$(CPP_HDRS))
endif
ifeq ($(GRID_BACKEND),none)
CPP_SRCS := $(filter-out $(CPP_SRCS_GRID),$(CPP_SRCS))
CPP_HDRS := $(filter-out $(CPP_HDRS_GRID),$(CPP_HDRS))
endif

CINT_SRCS := $(wildcard $(INTERFACEDIR)/Milhoja_*.cpp)
FINT_SRCS := $(wildcard $(INTERFACEDIR)/Milhoja_*.F90)
CINT_HDRS := $(wildcard $(INTERFACEDIR)/Milhoja_*.h)

ifeq      ($(RUNTIME_BACKEND),None)
CU_SRCS :=
CU_HDRS :=
CUFLAGS :=
else ifeq ($(RUNTIME_BACKEND),CUDA)
CXXFLAGS += -I$(INCDIR)/CudaBackend
CUFLAGS  = -I$(INCDIR) -I$(INCDIR)/CudaBackend -I$(BUILDDIR) \
           $(CUFLAGS_STD) $(CUFLAGS_PROD) $(CUFLAGS_AMREX)
CU_SRCS := $(wildcard $(SRCDIR)/CudaBackend/Milhoja_*.cu)
CU_HDRS := $(wildcard $(INCDIR)/CudaBackend/Milhoja_*.h)
else ifeq ($(RUNTIME_BACKEND),HOSTMEM)
CXXFLAGS += -I$(INCDIR)/FakeCudaBackend
CXXFLAGS += -I$(INCDIR) -I$(BUILDDIR)
CPP_SRCS += $(wildcard $(SRCDIR)/FakeCudaBackend/Milhoja_*.cpp)
CPP_HDRS += $(wildcard $(INCDIR)/FakeCudaBackend/Milhoja_*.h)
CU_SRCS :=
CU_HDRS :=
CUFLAGS :=
else ifeq ($(RUNTIME_BACKEND),OMPTARGET)
CXXFLAGS += -I$(INCDIR)/OmpTargetBackend
CXXFLAGS += -I$(INCDIR) -I$(BUILDDIR)
CPP_SRCS += $(wildcard $(SRCDIR)/OmpTargetBackend/Milhoja_*.cpp)
CPP_HDRS += $(wildcard $(INCDIR)/OmpTargetBackend/Milhoja_*.h)
CU_SRCS :=
CU_HDRS :=
CUFLAGS :=
else
$(error Unknown backend $(RUNTIME_BACKEND))
endif

CPP_OBJS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CPP_SRCS:.cpp=.o))
ifneq ($(BACKEND_BUILDDIR),)
CPP_OBJS := $(patsubst $(BACKEND_BUILDDIR)/%,$(BUILDDIR)/%,$(CPP_OBJS))
endif
INT_OBJS := $(patsubst $(INTERFACEDIR)/%,$(BUILDDIR)/%,$(CINT_SRCS:.cpp=.o))
INT_OBJS += $(patsubst $(INTERFACEDIR)/%,$(BUILDDIR)/%,$(FINT_SRCS:.F90=.o))
CU_OBJS  := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CU_SRCS:.cu=.o))
OBJS     := $(CPP_OBJS) $(INT_OBJS) $(CU_OBJS)
HDRS     := $(CPP_HDRS) $(CINT_HDRS) $(CU_HDRS)

ifeq      ($(COMPUTATION_OFFLOADING),None)
else ifeq ($(COMPUTATION_OFFLOADING),OpenACC)
CXXFLAGS += $(OACC_FLAGS)
else ifeq ($(COMPUTATION_OFFLOADING),OpenMP)
CXXFLAGS += $(OMP_FLAGS)
else
$(error Unknown computation offload $(COMPUTATION_OFFLOADING))
endif

.PHONY: all install clean
all:     $(TARGET) $(SIZES_JSON)
install:
	mkdir -pv $(LIB_MILHOJA_PREFIX) || exit $?
	test -d $(LIB_MILHOJA_PREFIX)/include && ls -l $(LIB_MILHOJA_PREFIX)/include
	mkdir -pv $(LIB_MILHOJA_PREFIX)/include || exit $?
	mkdir $(LIB_MILHOJA_PREFIX)/lib
	cp $(TARGET) $(LIB_MILHOJA_PREFIX)/lib
	cp $(MILHOJA_H) $(LIB_MILHOJA_PREFIX)/include
	cp $(HDRS) $(LIB_MILHOJA_PREFIX)/include
	cp $(BUILDDIR)/*.mod $(LIB_MILHOJA_PREFIX)/include
ifneq ($(SIZES_JSON),)
	cp $(SIZES_JSON) $(LIB_MILHOJA_PREFIX)/include
endif
	cp $(INTERFACEDIR)/*.finc $(LIB_MILHOJA_PREFIX)/include
clean:
	$(RM) $(BUILDDIR)/*.o
	$(RM) $(BUILDDIR)/*.d

$(BUILDDIR):
	@echo
	@echo "Intermediate build folder $(BUILDDIR) missing.  Created."
	@echo
	@mkdir -p $(BUILDDIR) $(BACKEND_BUILDDIR)

$(MILHOJA_H): $(MAKEFILES) | $(BUILDDIR)
	./tools/write_library_header.py --dim $(NDIM) \
                                     --runtime $(RUNTIME_BACKEND) \
                                     --grid $(GRID_BACKEND) \
                                     --fps $(FLOATING_POINT_SYSTEM) \
                                     --offload $(COMPUTATION_OFFLOADING) \
                                     $(MILHOJA_H) \
                                     $(if $(SUPPORT_EXEC),--support_exec) \
                                     $(if $(SUPPORT_PUSH),--support_push) \
                                     $(if $(SUPPORT_PACKETS),--support_packets)

# - Program depends directly on Milhoja.h and other Milhoja headers
# - Sizes might change if compiler flags are changed
$(SIZES_JSON): $(TOOLSDIR)/createSizesJson.cpp $(MILHOJA_H) $(CPP_HDRS) $(MAKEFILES)
	$(CXXCOMP) $(TOOLSDIR)/createSizesJson.cpp $(filter-out -c,$(CXXFLAGS)) -o $(BUILDDIR)/createSizesJson.x
	$(BUILDDIR)/createSizesJson.x $(SIZES_JSON)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(MILHOJA_H) $(HDRS) $(MAKEFILES)
	$(CXXCOMP) -c $(DEPFLAGS) $(CXXFLAGS) -o $@ $<

$(BUILDDIR)/%.o: $(INTERFACEDIR)/%.cpp $(MILHOJA_H) $(HDRS) $(MAKEFILES)
	$(CXXCOMP) -c $(DEPFLAGS) $(CXXFLAGS) -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu $(MILHOJA_H) $(HDRS) $(MAKEFILES)
	$(CUCOMP) -MM $(CUFLAGS) -o $(@:.o=.d) $<
	$(CUCOMP) -c $(CUFLAGS) -o $@ $<

ifeq ($(RUNTIME_BACKEND),HOSTMEM)
$(BUILDDIR)/Milhoja_FakeCuda%.o: $(SRCDIR)/FakeCudaBackend/Milhoja_FakeCuda%.cpp $(MILHOJA_H) $(HDRS) $(MAKEFILES)
	$(CXXCOMP) -c $(DEPFLAGS) $(CXXFLAGS) -o $@ $<
endif

ifeq ($(RUNTIME_BACKEND),OMPTARGET)
$(BUILDDIR)/Milhoja_OmpTarget%.o: $(SRCDIR)/OmpTargetBackend/Milhoja_OmpTarget%.cpp $(MILHOJA_H) $(HDRS) $(MAKEFILES)
	$(CXXCOMP) -c $(DEPFLAGS) $(CXXFLAGS) -o $@ $<
endif

# The build system does not have the facility to automatically discover
# dependencies between Fortran source files.  Since the only Fortran
# source files officially in the repo are in the high-level Fortran
# interface and these are few, we can manually manage dependencies
# and therefore write the build rules here.
$(BUILDDIR)/Milhoja_types_mod.o: $(INTERFACEDIR)/Milhoja_types_mod.F90 $(MILHOJA_H) $(MAKEFILES)
	$(F90COMP) -c $(F90FLAGS) -o $@ $<
$(BUILDDIR)/Milhoja_errors_mod.o: $(INTERFACEDIR)/Milhoja_errors_mod.F90 $(INTERFACEDIR)/Milhoja_interface_error_codes.h $(BUILDDIR)/Milhoja_types_mod.o Makefile
	$(F90COMP) -c $(F90FLAGS) -o $@ $<
$(BUILDDIR)/Milhoja_grid_mod.o: $(INTERFACEDIR)/Milhoja_grid_mod.F90 $(BUILDDIR)/Milhoja_runtime_mod.o $(BUILDDIR)/Milhoja_grid_C_interface.o $(MILHOJA_H) Makefile
	$(F90COMP) -c $(F90FLAGS) -o $@ $<
$(BUILDDIR)/Milhoja_tile_mod.o: $(INTERFACEDIR)/Milhoja_tile_mod.F90 $(BUILDDIR)/Milhoja_types_mod.o $(BUILDDIR)/Milhoja_tile_C_interface.o Makefile
	$(F90COMP) -c $(F90FLAGS) -o $@ $<
$(BUILDDIR)/Milhoja_runtime_mod.o: $(INTERFACEDIR)/Milhoja_runtime_mod.F90 $(BUILDDIR)/Milhoja_types_mod.o $(BUILDDIR)/Milhoja_runtime_C_interface.o Makefile
	$(F90COMP) -c $(F90FLAGS) -o $@ $<
$(BUILDDIR)/Milhoja_tileCInfo_mod.o: $(INTERFACEDIR)/Milhoja_tileCInfo_mod.F90 $(INTERFACEDIR)/Milhoja_tileCInfo.finc $(MILHOJA_H) Makefile
	$(F90COMP) -c $(F90FLAGS) -o $@ $<

$(TARGET): $(OBJS) Makefile
	ar -rcs $@ $(OBJS)

