SHELL=/bin/sh


########################################################
# Makefile flags and defintions

MAKEFILE     = Makefile
MAKEFILES    = $(MAKEFILE) Makefile.site Makefile.base Makefile.F_interface Makefile.F_rules $(if $(LIBONLY),,Makefile.test)

include Makefile.site
include Makefile.base
include Makefile.setup
include Makefile.F_interface
ifdef LIBONLY
else
include Makefile.test
endif

# Default shell commands
RM ?= /bin/rm

# Use C++11 standard, flags differ by compiler
# -MMD generates a dependecy list for each file as a side effect
ifeq ($(CXXCOMPNAME),gnu)
CXXFLAGS_STD = -std=c++11
DEPFLAG = -MMD
else ifeq ($(CXXCOMPNAME), intel)
# With 20.2, I get lots of warnings related to AMReX if
# I limit the code to c++11.  They are programming to a more
# modern standard.
CXXFLAGS_STD =
DEPFLAG = -MMD
else ifeq ($(CXXCOMPNAME), pgi)
CXXFLAGS_STD = -std=c++11
DEPFLAG = -MMD
else ifeq ($(CXXCOMPNAME), ibm)
CXXFLAGS_STD = -std=c++11
DEPFLAG = -MMD
else ifeq ($(CXXCOMPNAME), llvm)
CXXFLAGS_STD = -std=c++11
DEPFLAG = -MMD
else
$(info $(CXXCOMPNAME) compiler not yet supported.)
endif
CUFLAGS_STD  = -std=c++11

CXXFLAGS_LIB =
F90FLAGS_LIB =
ifdef LIBONLY
# TODO: Is this necessary?  Should it be -fpic?  What about -fpie/fPIE for
# tests?
CXXFLAGS_LIB = -fPIC
F90FLAGS_LIB = -fPIC
endif

# Combine all compiler and linker flags
ifeq ($(DEBUG),true)
CXXFLAGS = $(CXXFLAGS_STD) $(CXXFLAGS_DEBUG) -I$(BUILDDIR) $(CXXFLAGS_BASE) \
           $(CXXFLAGS_LIB) $(CXXFLAGS_TEST_DEBUG) $(CXXFLAGS_AMREX)
F90FLAGS = $(F90FLAGS_DEBUG) $(F90FLAGS_INTERFACE) $(F90FLAGS_LIB)
else
CXXFLAGS = $(CXXFLAGS_STD) $(CXXFLAGS_PROD) -I$(BUILDDIR) $(CXXFLAGS_BASE) \
           $(CXXFLAGS_LIB) $(CXXFLAGS_TEST_PROD) $(CXXFLAGS_AMREX)
F90FLAGS = $(F90FLAGS_PROD) $(F90FLAGS_INTERFACE) $(F90FLAGS_LIB)
endif
CUFLAGS  = $(CUFLAGS_STD) $(CUFLAGS_PROD) $(CUFLAGS_BASE) $(CUFLAGS_TEST) \
	   $(CUFLAGS_AMREX) -I$(BUILDDIR)
LDFLAGS  = -L$(LIB_RUNTIME) -lruntime $(LIB_AMREX) $(LDFLAGS_TEST) $(LDFLAGS_STD)


# Add code coverage flags
ifeq ($(CODECOVERAGE), true)
CXXFLAGS += $(CXXFLAGS_COV)
LDFLAGS  += $(LDFLAGS_COV)
endif

# Adjust flags for multithreaded distributor
ifeq ($(THREADED_DISTRIBUTOR),true)
$(info Warning! multi-threaded distributor not fully tested)
AMREXDIR     = $(AMREXDIR_OMP)
CXXFLAGS    += $(OMP_FLAGS) -DUSE_THREADED_DISTRIBUTOR
CUFLAGS     += $(CU_OMP_FLAGS)
endif


# List of sources, objects, and dependencies
C_SRCS    = $(SRCS_BASE) $(SRCS_TEST) $(C_SRCS_INTERFACE)
SRCS      = $(C_SRCS) $(CU_SRCS)

C_OBJS    = $(addsuffix .o, $(basename $(notdir $(C_SRCS))))
CU_OBJS   = $(addsuffix .o, $(basename $(notdir $(CU_SRCS))))
OBJS      = $(C_OBJS) $(CU_OBJS)
DEPS      = $(OBJS:.o=.d)

OBJS_TEST      = $(CU_OBJS) $(addsuffix .o, $(basename $(notdir $(SRCS_TEST))))
OBJS_BASE      = $(addsuffix .o, $(basename $(notdir $(SRCS_BASE))))
OBJS_INTERFACE = \
	$(addsuffix .o, $(basename $(notdir $(C_SRCS_INTERFACE)))) \
	$(addsuffix .o, $(basename $(notdir $(F_SRCS_INTERFACE))))

# Use vpath as suggested here: http://make.mad-scientist.net/papers/multi-architecture-builds/#single
# This allows all targets to be put a single directory (the build directory) and directs the Makefile to
# search the source tree for the prerequisites.
vpath %.cpp $(sort $(dir $(C_SRCS)))
vpath %.cu  $(sort $(dir $(CU_SRCS)))
vpath %.F90 $(sort $(dir $(F_SRCS_INTERFACE)))

##########################################################
# Makefile commands:

.PHONY: default all clean test install
default: $(if $(LIBONLY), libruntime.a, $(BINARYNAME))
all:     $(if $(LIBONLY), libruntime.a, $(BINARYNAME))
test:
ifdef LIBONLY
else
	./$(BINARYNAME)
endif


# If code coverage is being build into the test, remove any previous gcda files to avoid conflict.
$(BINARYNAME): $(OBJS_TEST) $(MAKEFILES) $(if $(LINKLIB), ,libruntime.a)
ifeq ($(CODECOVERAGE), true)
	$(RM) -f *.gcda
endif
	$(CXXCOMP) -o $(BINARYNAME) $(OBJS_TEST) $(LDFLAGS)

%.o: %.cpp $(MAKEFILES)
	$(CXXCOMP) -c $(DEPFLAG) $(CXXFLAGS) -o $@ $<

%.o: %.cu $(MAKEFILES)
	$(CUCOMP) -MM $(CUFLAGS) -o $(@:.o=.d) $<
	$(CUCOMP) -c $(CUFLAGS) -o $@ $<

# Since automatic discover of dependencies between Fortran files is not yet
# included in the build system, allow the interface portion of the repo to
# specify its own build rules with dependencies manually specified.  This is
# acceptable since the number of Fortran files there is small.
include Makefile.F_rules

# Commands for just compiling the Runtime into a library
runtime: libruntime.a

libruntime.a: $(OBJS_BASE) $(OBJS_INTERFACE) $(MAKEFILES)
	ar -rcs $@ $(OBJS_BASE) $(OBJS_INTERFACE)

install:
ifdef LIBONLY
	mkdir -p $(BASEDIR)/$(LIB_RUNTIME_PREFIX)
	cp libruntime.a $(BASEDIR)/$(LIB_RUNTIME_PREFIX)/
	cp *.mod $(BASEDIR)/$(LIB_RUNTIME_PREFIX)/
	cp $(INTERFACEDIR)/milhoja_interface_error_codes.h $(BASEDIR)/$(LIB_RUNTIME_PREFIX)/
endif


# Clean removes all intermediate files
clean:
	$(RM) -f *.o
	$(RM) -f *.d
	$(RM) -f *.a
ifeq ($(CODECOVERAGE), true)
	$(RM) -f *.gcno
	$(RM) -f *.gcda
endif
	$(RM) -f lcov_temp.info


.PHONY: coverage
coverage:
ifeq ($(CODECOVERAGE), true)
	$(LCOV) -o lcov_temp.info -c -d .
	$(GENHTML)  -o Coverage_Report lcov_temp.info
else
	$(info Include --coverage in your setup line to enable code coverage.)
endif


# Include dependencies generated by compiler
-include $(DEPS)

