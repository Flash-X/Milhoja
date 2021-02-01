SHELL=/bin/sh

########################################################
# Makefile flags and defintions

MAKEFILE     = Makefile

#TODO: how to deal with this compiler specific flag
CXXFLAGS = -std=c++11
CXXWARNS =

LIBS =
LDFLAGS =

# TODO Include these for only if code coverage requested
CXXFLAGS += -fprofile-arcs -ftest-coverage
LDFLAGS += --coverage


# Makefile.site defines BASEDIR, AMREXDIR and GTESTDIR, and CXXCOMP
include Makefile.site
# Makefile.base defines SRCS and INCDIR
include Makefile.base
# Makefile.test defines BINARYNAME, TESTDIR and adds to SRCS
include Makefile.test

# List of object files
OBJS        = $(addsuffix .o, $(basename $(SRCS)))
# List of object files without full path
OBJSINBUILD = $(addsuffix .o, $(basename $(notdir $(SRCS))))


# TODO: is this needed?
ifdef DEBUG
CXXFLAGS += -DDEBUG_RUNTIME
endif
# TODO replace this with a better verbosity system
CXXFLAGS += -DGRID_LOG

##########################################################
# Makefile commands:

.PHONY: default all clean
default: $(BINARYNAME)
all:     $(BINARYNAME)
test:
	./$(BINARYNAME)


# Main make command depends on making all object files first
# TODO investigate if dependencies actually work here
$(BINARYNAME): $(OBJS) $(MAKEFILE) Makefile.site Makefile.base Makefile.test
	$(CXXCOMP) -o $(BINARYNAME) $(OBJSINBUILD) $(LDFLAGS) $(LIBS)

# $(notdir $@) puts object files in build dir
%.o: %.cpp $(CXX_HDRS) $(MAKEFILE) Makefile.site Makefile.base Makefile.test
	$(CXXCOMP) -c $(CXXFLAGS) $(CXXWARNS) -o $(notdir $@) $<

# TODO: Only do if code coverage requested
.PHONY: default all clean lcov
lcov:
	$(COVERAGETOOL) -o lcov_temp.info -c -d .
	$(HTMLCOMMAND)  -o Coverage_Report lcov_temp.info

clean:
	/bin/rm -f $(BINARYNAME)
	/bin/rm -f *.log
	/bin/rm -f *.o
	/bin/rm -f *.gcda
	/bin/rm -f *.gcno
	/bin/rm -f lcov_temp.info

