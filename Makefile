SHELL=/bin/sh

########################################################
# Makefile flags and defintions

BASEDIR      = ..
MAKEFILE     = Makefile
INCDIR       = $(BASEDIR)/includes
TESTBASEDIR  = $(BASEDIR)/test/Base

# Makefile.site defines in AMREXDIR and GTESTDIR
include Makefile.site
# Makefile.base defines SRCS
include Makefile.base
# Makefile.test defines BINARYNAME, TESTDIR and adds to SRCS
include Makefile.test

# List of object files
OBJS        = $(addsuffix .o, $(basename $(SRCS)))
# List of object files without full path
OBJSINBUILD = $(addsuffix .o, $(basename $(notdir $(SRCS))))

# Flags for compiling cpp files
CXXCOMP   = mpicxx
CXXFLAGS  = -g -O0 -std=c++11 -I$(INCDIR) -I$(TESTBASEDIR) -I$(TESTDIR) -I$(AMREXDIR)/include -I$(GTESTDIR)/include
CXXWARNS  =
# TODO Include these for only if code coverage requested
CXXFLAGS += -fprofile-arcs -ftest-coverage

# Linker flags
LIBS      = -lpthread -lstdc++ -lamrex -lgtest
LDFLAGS   = -L$(AMREXDIR)/lib -L$(GTESTDIR)/mybuild/lib
# TODO  For code coverage only
LDFLAGS += --coverage

# TODO: is this needed?
ifdef DEBUG
CXXFLAGS := $(CXXFLAGS) -DDEBUG_RUNTIME
endif
# TODO replace this with a better verbosity system
CXXFLAGS += -DGRID_LOG

##########################################################
# Makefile commands:

.PHONY: default all clean
default: $(BINARYNAME)
all:     $(BINARYNAME)

# Main make command depends on making all object files first
$(BINARYNAME): $(OBJS) $(MAKEFILE)
	$(CXXCOMP) -o $(BINARYNAME) $(OBJSINBUILD) $(LDFLAGS) $(LIBS)

%.o: %.cpp $(CXX_HDRS) $(MAKEFILE)
	# $(notdir $@) puts object files in build dir
	$(CXXCOMP) -c $(CXXFLAGS) $(CXXWARNS) -o $(notdir $@) $<

# TODO: Only do if code coverage requested
.PHONY: lcov
lcov:
	lcov -o lcov_temp.info -c -d .
	genhtml -o Coverage_Report lcov_temp.info

clean:
	/bin/rm -f $(BINARYNAME)
	/bin/rm -f *.log
	/bin/rm -f *.o
	/bin/rm -f *.gcda
	/bin/rm -f *.gcno

