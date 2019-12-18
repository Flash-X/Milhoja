SHELL=/bin/sh

BASE    = test_runtime
BASEDIR = .
INCDIR  = $(BASEDIR)/includes
SRCDIR  = $(BASEDIR)/src

# Common files
CXX_HDRS   = \
    $(INCDIR)/runtimeTask.h \
    $(INCDIR)/cpuThreadRoutine.h \
    $(INCDIR)/gpuThreadRoutine.h \
    $(INCDIR)/postGpuThreadRoutine.h \
    $(INCDIR)/ThreadTeam.h \
    $(INCDIR)/OrchestrationRuntime.h
SRCS       = \
    $(SRCDIR)/cpuThreadRoutine.cpp \
    $(SRCDIR)/gpuThreadRoutine.cpp \
    $(SRCDIR)/postGpuThreadRoutine.cpp \
    $(SRCDIR)/ThreadTeam.cpp \
    $(SRCDIR)/OrchestrationRuntime.cpp \
    $(SRCDIR)/Driver.cpp

OBJS      = $(addsuffix .o, $(basename $(SRCS)))
MAKEFILE  = Makefile
COMMAND   =  $(BASE).x

CXX       = g++
CXXFLAGS  = -fopenmp -g -O0 -I$(OMPI_DIR)/include -I$(INCDIR) -std=c++11
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
 
