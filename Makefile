SHELL=/bin/sh

##### MIMIC SETUP STAGE/RUNTIME PARAMETERS HERE
# These define default values, but selection can also be made at
# command line.
#
# Ex. The full featured test of the Orchestration System prototype
#    make clean all RUNTIME=CUDA VERBOSITY=SILENT
#

#
# Choose between 
# - running a serial invocation (CPU-only node operation) or
# - running a parallel invocation (CPU/Multi-GPU node operation)
#
ifndef RUNTIME
RUNTIME=CPU_ONLY
#RUNTIME=CUDA
endif

#
# Enable/Disable verbose logging of orchestration runtime sequence
#
ifndef VERBOSITY
VERBOSITY=VERBOSE
#VERBOSITY=SILENT
endif

BASE     = test_runtime
BASEDIR  = .
INCDIR   = $(BASEDIR)/includes
SRCDIR   = $(BASEDIR)/src
TESTDIR  = $(BASEDIR)/test
AMREXDIR = $(HOME)/Projects/amrex_install/2D
GTESTDIR = /usr/local/spack/opt/spack/darwin-highsierra-x86_64/gcc-6.5.0/googletest-1.8.1-6y5spjfzq4uiazxbcb7sw3qrloxbrhgw

# Common files
CXX_HDRS   = \
    $(INCDIR)/runtimeTask.h \
    $(INCDIR)/ThreadTeamModes.h \
    $(INCDIR)/ThreadTeam.h \
    $(INCDIR)/ThreadTeamState.h \
    $(INCDIR)/ThreadTeamIdle.h \
    $(INCDIR)/ThreadTeamTerminating.h \
    $(INCDIR)/ThreadTeamRunningOpen.h \
    $(INCDIR)/ThreadTeamRunningClosed.h \
    $(INCDIR)/ThreadTeamRunningNoMoreWork.h \
    $(TESTDIR)/constants.h
#    $(INCDIR)/OrchestrationRuntime.h \
#    $(TESTDIR)/computeLaplacianDensity_cpu.h \
#    $(TESTDIR)/computeLaplacianEnergy_cpu.h \
#    $(TESTDIR)/scaleEnergy_cpu.h
SRCS       = \
    $(TESTDIR)/testThreadRoutines.cpp \
    $(TESTDIR)/testRuntimeInt.cpp \
    $(TESTDIR)/testRuntimeBlock.cpp
#    $(TESTDIR)/testThreadTeam.cpp \
#    $(TESTDIR)/computeLaplacianDensity_cpu.cpp \
#    $(TESTDIR)/computeLaplacianEnergy_cpu.cpp \
#    $(TESTDIR)/scaleEnergy_cpu.cpp \

OBJS      = $(addsuffix .o, $(basename $(SRCS)))
MAKEFILE  = Makefile
COMMAND   =  $(BASE).x

CXX       = mpicxx
CXXFLAGS  = -g -O0 -I$(INCDIR) -I$(TESTDIR) -I$(AMREXDIR)/include -I$(GTESTDIR)/include -std=c++11 -D$(RUNTIME) -D$(VERBOSITY)
CXXWARNS  =

LIBS      = -lstdc++ -lamrex -lgtest -lgtest_main
LDFLAGS   = -L$(AMREXDIR)/lib -L$(GTESTDIR)/lib
 
all:    $(COMMAND)

.SUFFIXES:
.SUFFIXES: .o .cpp

$(COMMAND): $(OBJS) $(MAKEFILE) 
	$(CXX) -o $(COMMAND) $(OBJS) $(LDFLAGS) $(LIBS)

.cpp.o: $(CXX_HDRS) $(MAKEFILE)
	$(CXX) -c $(CXXFLAGS) $(CXXWARNS) -o $@ $<

clean:
	/bin/rm -f $(COMMAND) $(BASEDIR)/Test*.log
	/bin/rm -f $(COMMAND) $(SRCDIR)/*.o
	/bin/rm -f $(COMMAND) $(TESTDIR)/*.o

test: $(BASEDIR)/$(BASE).x
	@echo
	@echo "Run Orchestration Runtime testsuites"
	@echo "------------------------------------------------------------------"
	@$(BASEDIR)/$(BASE).x
	@echo

