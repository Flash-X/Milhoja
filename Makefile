SHELL=/bin/sh


########################################################
# Makefile flags and defintions

MAKEFILE     = Makefile

# Must include site first so BASEDIR is defined.
include Makefile.site
include Makefile.base
include Makefile.test

#TODO: how to deal with this compiler specific flag
CXXFLAGS_STD = -std=c++11
LDFLAGS_STD  = -lpthread -lstdc++

# TODO Include these for only if code coverage requested
CXXFLAGS_COV = -fprofile-arcs -ftest-coverage
LDFLAGS_COV  = --coverage

# Combine all compiler and linker flags
CXXFLAGS = $(CXXFLAGS_STD) $(CXXFLAGS_PROD) $(CXXFLAGS_BASE) $(CXXFLAGS_TEST) \
           $(CXXFLAGS_AMREX) $(CXXFLAGS_COV)
LDFLAGS  = $(LDFLAGS_STD) $(LIB_AMREX) $(LDFLAGS_TEST) $(LDFLAGS_COV)

# List of object files
SRCS     = $(SRCS_BASE) $(SRCS_TEST)
OBJS     = $(addsuffix .o, $(basename $(SRCS)))
## List of object files without full path
#OBJSINBUILD = $(addsuffix .o, $(basename $(notdir $(SRCS))))


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
	/bin/rm -f $(SRCDIR)/*.gcda
	/bin/rm -f $(TESTDIR)/*.gcda
	/bin/rm -f $(TESTBASEDIR)/*.gcda
	$(CXXCOMP) -o $(BINARYNAME) $(OBJS) $(LDFLAGS)

%.o: %.cpp $(CXX_HDRS) $(MAKEFILE) Makefile.site Makefile.base Makefile.test
	$(CXXCOMP) -c $(CXXFLAGS) $(CXXWARNS) -o $@ $<

# TODO: Only do if code coverage requested, fail if test not run
.PHONY: default all clean coverage
coverage:
	$(LCOV) -o lcov_temp.info -c -d $(BASEDIR)
	$(GENHTML)  -o Coverage_Report lcov_temp.info

clean:
	/bin/rm -f $(BINARYNAME)
	/bin/rm -f *.log
	/bin/rm -f $(SRCDIR)/*.o
	/bin/rm -f $(TESTDIR)/*.o
	/bin/rm -f $(TESTBASEDIR)/*.o
	/bin/rm -f $(SRCDIR)/*.gcno
	/bin/rm -f $(TESTDIR)/*.gcno
	/bin/rm -f $(TESTBASEDIR)/*.gcno
	/bin/rm -f $(SRCDIR)/*.gcda
	/bin/rm -f $(TESTDIR)/*.gcda
	/bin/rm -f $(TESTBASEDIR)/*.gcda
	/bin/rm -f *.gcda
	/bin/rm -f *.gcno
	/bin/rm -f lcov_temp.info

