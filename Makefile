SHELL=/bin/sh

##### MIMIC SETUP STAGE/RUNTIME PARAMETERS HERE
# These define default values, but selection can also be made at
# command line.
#
# Ex. The full featured test of the Orchestration System prototype
#    make clean all RUNTIME=CUDA VERBOSITY=SILENT STUDY=SCALING
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
#VERBOSITY=VERBOSE
VERBOSITY=SILENT
endif

#
# Execute the test on 
# - a single Grid setup for rapid regression testing or
# - a series of increasingly refined meshes as a scaling test 
#
ifndef STUDY
STUDY=SINGLE
#STUDY=SCALING
endif

BASE     = test_runtime
BASEDIR  = .
INCDIR   = $(BASEDIR)/includes
SRCDIR   = $(BASEDIR)/src
TESTDIR  = $(BASEDIR)/test

# Common files
CXX_HDRS   = \
    $(INCDIR)/constants.h \
    $(INCDIR)/Block.h \
    $(INCDIR)/Grid.h \
    $(INCDIR)/BlockIterator.h \
    $(INCDIR)/runtimeTask.h \
    $(INCDIR)/computeLaplacianDensity_cpu.h \
    $(INCDIR)/computeLaplacianEnergy_cpu.h \
    $(INCDIR)/scaleEnergy_cpu.h \
    $(INCDIR)/ThreadTeam.h \
    $(INCDIR)/ThreadTeamState.h \
    $(INCDIR)/ThreadTeamIdle.h \
    $(INCDIR)/ThreadTeamTerminating.h \
    $(INCDIR)/ThreadTeamRunningOpen.h \
    $(INCDIR)/ThreadTeamRunningClosed.h \
    $(INCDIR)/ThreadTeamRunningNoMoreWork.h \
    $(INCDIR)/OrchestrationRuntime.h
SRCS       = \
    $(SRCDIR)/Block.cpp \
    $(SRCDIR)/Grid.cpp \
    $(SRCDIR)/BlockIterator.cpp \
    $(SRCDIR)/computeLaplacianDensity_cpu.cpp \
    $(SRCDIR)/computeLaplacianEnergy_cpu.cpp \
    $(SRCDIR)/scaleEnergy_cpu.cpp \
    $(SRCDIR)/ThreadTeam.cpp \
    $(SRCDIR)/ThreadTeamIdle.cpp \
    $(SRCDIR)/ThreadTeamTerminating.cpp \
    $(SRCDIR)/ThreadTeamRunningOpen.cpp \
    $(SRCDIR)/ThreadTeamRunningClosed.cpp \
    $(SRCDIR)/ThreadTeamRunningNoMoreWork.cpp \
    $(SRCDIR)/OrchestrationRuntime.cpp \
    $(TESTDIR)/testThreadRoutines.cpp \
    $(TESTDIR)/test.cpp

OBJS      = $(addsuffix .o, $(basename $(SRCS)))
MAKEFILE  = Makefile
COMMAND   =  $(BASE).x

GTESTDIR = /usr/local/spack/opt/spack/darwin-highsierra-x86_64/gcc-6.5.0/googletest-1.8.1-4fb34iawhssxssc3mdpe4cjjldgnr6n7
CXX       = g++
CXXFLAGS  = -g -O0 -I$(INCDIR) -I$(TESTDIR) -I$(GTESTDIR)/include -std=c++11 -D$(RUNTIME) -D$(VERBOSITY) -D$(STUDY) 
CXXWARNS  =

LIBS      = -lstdc++ -lgtest -lgtest_main
LDFLAGS   = -L$(GTESTDIR)/lib
 
all:    $(COMMAND)

.SUFFIXES:
.SUFFIXES: .o .cpp

$(COMMAND): $(OBJS) $(MAKEFILE) 
	$(CXX) -o $(COMMAND) $(OBJS) $(LDFLAGS) $(LIBS)

.cpp.o: $(CXX_HDRS) $(MAKEFILE)
	$(CXX) -c $(CXXFLAGS) $(CXXWARNS) -o $@ $<

clean:
	/bin/rm -f $(COMMAND) $(SRCDIR)/*.o

test: $(BASEDIR)/$(BASE).x
	@echo
	@echo "Run Orchestration Runtime testsuites"
	@echo "------------------------------------------------------------------"
	@$(BASEDIR)/$(BASE).x
	@echo

