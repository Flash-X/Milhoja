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
ifeq ($(CXXCOMPNAME),gnu)
CXXFLAGS_STD = -std=c++11
else
endif


# Combine all compiler and linker flags
ifeq ($(DEBUG),true)
CXXFLAGS = $(CXXFLAGS_STD) $(CXXFLAGS_DEBUG) $(CXXFLAGS_BASE) $(CXXFLAGS_TEST) \
           $(CXXFLAGS_AMREX) -I$(BUILDDIR)
LDFLAGS  = $(LDFLAGS_STD) $(LIB_AMREX) $(LDFLAGS_TEST)
else
CXXFLAGS = $(CXXFLAGS_STD) $(CXXFLAGS_PROD) $(CXXFLAGS_BASE) $(CXXFLAGS_TEST) \
           $(CXXFLAGS_AMREX) -I$(BUILDDIR)
LDFLAGS  = $(LIB_AMREX) $(LDFLAGS_TEST) $(LDFLAGS_STD) 
endif


# Add code coverage flags
ifeq ($(CODECOVERAGE), true)
CXXFLAGS += $(CXXFLAGS_COV)
LDFLAGS  += $(LDFLAGS_COV)
endif

# List of sources, objects, and dependencies
C_SRCS     = $(SRCS_BASE) $(SRCS_TEST)
OBJS_TEMP     = $(addsuffix .o, $(basename $(C_SRCS)))
OBJS     =  $(patsubst $(BASEDIR)/%,$(OBJDIR)/%,$(OBJS_TEMP))
OBJTREE  =  $(sort $(dir $(OBJS)))
DEPS     = $(addsuffix .d, $(basename $(OBJS)))


# TODO: is this needed?
ifeq ($(DEBUG), true)
CXXFLAGS += -DDEBUG_RUNTIME
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
	/bin/rm -f $(addsuffix *.gcda,$(OBJTREE))
endif
	$(CXXCOMP) -o $(BINARYNAME) $(OBJS) $(LDFLAGS)

# -MMD generates a dependecy list for each file as a side effect
$(OBJDIR)/%.o: $(BASEDIR)/%.cpp  $(MAKEFILES) | $(OBJTREE)
	$(CXXCOMP) -MMD -c $(CXXFLAGS) -o $@ $<

# Make directories in the object tree
$(OBJTREE):
	mkdir -p $@

# Clean removes all intermediate files
clean:
	/bin/rm -f $(addsuffix *.o,$(OBJTREE))
	/bin/rm -f $(addsuffix *.d,$(OBJTREE))
ifeq ($(CODECOVERAGE), true)
	/bin/rm -f $(addsuffix *.gcno,$(OBJTREE))
	/bin/rm -f $(addsuffix *.gcda,$(OBJTREE))
endif
	/bin/rm -f lcov_temp.info


.PHONY: coverage
coverage:
ifeq ($(CODECOVERAGE), true)
	$(LCOV) -o lcov_temp.info -c -d $(OBJDIR)
	$(GENHTML)  -o Coverage_Report lcov_temp.info
else
	$(info Include --coverage in your setup line to enable code coverage.)
endif


# Include dependencies generated by gcc
-include $(DEPS)

