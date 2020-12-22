SHELL=/bin/sh

BASEDIR      = ..
MAKEFILE     = Makefile
INCDIR       = $(BASEDIR)/includes
include Makefile.site

include Makefile.base
include Makefile.test

OBJS      = $(addsuffix .o, $(basename $(SRCS)))
COMMAND   =  $(BASE).x

CXX       = mpicxx
CXXFLAGS  = -g -O0 -std=c++11 -I$(INCDIR) -I$(TESTBASEDIR) -I$(TESTDIR) -I$(AMREXDIR)/include -I$(GTESTDIR)/include
CXXWARNS  =

#LIBS      = -lpthread -lstdc++ -lgfortran -lamrex -lgtest
LIBS      = -lpthread -lstdc++ -lamrex -lgtest
LDFLAGS   = -L$(AMREXDIR)/lib -L$(GTESTDIR)/mybuild/lib

ifdef DEBUG
CXXFLAGS := $(CXXFLAGS) -DDEBUG_RUNTIME
endif
ifdef GLOG
CXXFLAGS := $(CXXFLAGS) -DGRID_LOG
endif

all:    $(COMMAND)

.SUFFIXES:
.SUFFIXES: .o .cpp

$(COMMAND): $(OBJS) $(MAKEFILE) 
	$(CXX) -o $(COMMAND) $(OBJS) $(LDFLAGS) $(LIBS)

.cpp.o: $(CXX_HDRS) $(MAKEFILE)
	$(CXX) -c $(CXXFLAGS) $(CXXWARNS) -o $@ $<

clean:
	/bin/rm -f $(COMMAND)
	/bin/rm -f Test*.log
	/bin/rm -f $(SRCDIR)/*.o
	/bin/rm -f $(TESTBASEDIR)/*.o
	/bin/rm -f $(TESTDIR)/*.o

