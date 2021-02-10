SHELL=/bin/sh


########################################################
# Makefile flags and defintions

MAKEFILE     = Makefile
MAKEFILES    = $(MAKEFILE) Makefile.site Makefile.base Makefile.test

# Must include site first so BASEDIR is defined.
include Makefile.site
include Makefile.base
include Makefile.test
include Makefile.setup

# Use C++11 standard, flags differ by compiler
# -MMD generates a dependecy list for each file as a side effect
ifeq ($(CXXCOMPNAME),gnu)
CXXFLAGS_STD = -std=c++11
DEPFLAG = -MMD
else ifeq ($(CXXCOMPNAME), pgi)
CXXFLAGS_STD = -std=c++11
DEPFLAG = -MMD
else
$(info $(CXXCOMPNAME) compiler not yet supported.)
endif
CUFLAGS_STD  = -std=c++11


# Combine all compiler and linker flags
ifeq ($(DEBUG),true)
CXXFLAGS = $(CXXFLAGS_STD) $(CXXFLAGS_DEBUG) $(CXXFLAGS_BASE) $(CXXFLAGS_TEST) \
           $(CXXFLAGS_AMREX) -I$(BUILDDIR)
else
CXXFLAGS = $(CXXFLAGS_STD) $(CXXFLAGS_PROD) $(CXXFLAGS_BASE) $(CXXFLAGS_TEST) \
           $(CXXFLAGS_AMREX) -I$(BUILDDIR)
endif
CUFLAGS  = $(CUFLAGS_STD) $(CUFLAGS_PROD) $(CUFLAGS_BASE) $(CUFLAGS_TEST) \
	   $(CUFLAGS_AMREX) -I$(BUILDDIR)
LDFLAGS  = $(LDFLAGS_STD) $(LIB_AMREX) $(LDFLAGS_TEST)


# Add code coverage flags
ifeq ($(CODECOVERAGE), true)
CXXFLAGS += $(CXXFLAGS_COV)
LDFLAGS  += $(LDFLAGS_COV)
endif

# Adjust flags for multithreaded distributor
ifeq ($(THREADED_DISTRIBUTOR),true)
$(info Warning! multi-threaded distributor not tested yet)
AMREXDIR     = $(AMREXDIR_OMP)
CXXFLAGS    += $(OMP_FLAGS) -DUSE_THREADED_DISTRIBUTOR
CUFLAGS     += $(CU_OMP_FLAGS)
endif


# List of sources, objects, and dependencies
C_SRCS    = $(SRCS_BASE) $(SRCS_TEST)
SRCS      = $(C_SRCS) $(CU_SRCS)

C_OBJS    = $(addsuffix .o, $(basename $(notdir $(C_SRCS))))
CU_OBJS   = $(addsuffix .o, $(basename $(notdir $(CU_SRCS))))
OBJS      = $(C_OBJS) $(CU_OBJS)
DEPS      = $(OBJS:.o=.d)

# Use VPATH as suggested here: http://make.mad-scientist.net/papers/multi-architecture-builds/#single
# This puts all targets in a single directory (the build directory) and allows the Makefile to
# search the source tree for the prerequisites.
VPATH    = $(sort $(dir $(C_SRCS)))


# TODO: is this needed?
ifeq ($(DEBUG), true)
#CXXFLAGS += -DDEBUG_RUNTIME
endif

##########################################################
# Makefile commands:

.PHONY: default all clean test
default: $(BINARYNAME)
all:     $(BINARYNAME)
test:
	./$(BINARYNAME)


# Main make command depends on making all object files and creating object tree
# If code coverage is being build into the test, remove any previous gcda files to avoid conflict.
$(BINARYNAME): $(OBJS) $(MAKEFILES)
ifeq ($(CODECOVERAGE), true)
	/bin/rm -f *.gcda
endif
	$(CXXCOMP) -o $(BINARYNAME) $(OBJS) $(LDFLAGS)

%.o: %.cpp $(MAKEFILES)
	$(CXXCOMP) -c $(DEPFLAG) $(CXXFLAGS) -o $@ $<

%.o: %.cu $(MAKEFILES)
	$(CUCOMP) -MM $(CUFLAGS) -o $(@:.o=.d) $<
	$(CUCOMP) -c $(CUFLAGS) -o $@ $<


# Clean removes all intermediate files
clean:
	/bin/rm -f *.o
	/bin/rm -f *.d
ifeq ($(CODECOVERAGE), true)
	/bin/rm -f *.gcno
	/bin/rm -f *.gcda
endif
	/bin/rm -f lcov_temp.info


.PHONY: coverage
coverage:
ifeq ($(CODECOVERAGE), true)
	$(LCOV) -o lcov_temp.info -c -d .
	$(GENHTML)  -o Coverage_Report lcov_temp.info
else
	$(info Include --coverage in your setup line to enable code coverage.)
endif


# Include dependencies generated by gcc
-include $(DEPS)

