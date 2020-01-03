SHELL=/bin/sh

##### MIMIC SETUP STAGE/RUNTIME PARAMETERS HERE
# These define default values, but selection can also be made at
# command line.
#
# Ex. The full featured test of the Orchestration System prototype
#    make clean all RUNTIME=CUDA VERBOSITY=SILENT PROBLEM=P2 STUDY=SCALING
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

#
# Choose between two different problems to solve (See Driver.cpp)
#
ifndef PROBLEM
#PROBLEM=P1
PROBLEM=P2
endif

#
# Execute the test on 
# - a single Grid setup for rapid regression testing or
# - a series of increasingly refined meshes as a scaling test 
#
ifndef STUDY
#STUDY=SINGLE
STUDY=SCALING
endif

BASE    = test_runtime
BASEDIR = .
INCDIR  = $(BASEDIR)/includes
SRCDIR  = $(BASEDIR)/src

# Common files
CXX_HDRS   = \
    $(INCDIR)/constants.h \
    $(INCDIR)/Block.h \
    $(INCDIR)/Grid.h \
    $(INCDIR)/BlockIterator.h \
    $(INCDIR)/runtimeTask.h \
    $(INCDIR)/computeLaplacian_cpu.h \
    $(INCDIR)/scale_cpu.h \
    $(INCDIR)/ThreadTeam.h \
    $(INCDIR)/OrchestrationRuntime.h
SRCS       = \
    $(SRCDIR)/Block.cpp \
    $(SRCDIR)/Grid.cpp \
    $(SRCDIR)/BlockIterator.cpp \
    $(SRCDIR)/computeLaplacian_cpu.cpp \
    $(SRCDIR)/scale_cpu.cpp \
    $(SRCDIR)/ThreadTeam.cpp \
    $(SRCDIR)/OrchestrationRuntime.cpp \
    $(SRCDIR)/Driver.cpp

OBJS      = $(addsuffix .o, $(basename $(SRCS)))
MAKEFILE  = Makefile
COMMAND   =  $(BASE).x

CXX       = g++
CXXFLAGS  = -fopenmp -g -O0 -I$(INCDIR) -std=c++11 -D$(RUNTIME) -D$(VERBOSITY) -D$(PROBLEM) -D$(STUDY) 
CXXWARNS  =

LIBS      = -lgomp -lstdc++
LDFLAGS   = 
 
all:    $(COMMAND)

.SUFFIXES:
.SUFFIXES: .o .cpp

$(COMMAND): $(OBJS) $(MAKEFILE) 
	$(CXX) -o $(COMMAND) $(OBJS) $(LDFLAGS) $(LIBS)

.cpp.o: $(CXX_HDRS) $(MAKEFILE)
	$(CXX) -c $(CXXFLAGS) $(CXXWARNS) -o $@ $<

clean:
	/bin/rm -f $(COMMAND) $(SRCDIR)/*.o
 
