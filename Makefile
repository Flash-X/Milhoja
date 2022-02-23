# It is intended that make be run from the same folder as the Makefile.  All
# executions of make that build the library should start from a clean build.
#TODO: Add in full use of dependency files carefully.
#TODO: Make Makefile.setup here?

SHELL=/bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .o

INCDIR   = ./includes
SRCDIR   = ./src
BUILDDIR = ./build
TARGET   = $(BUILDDIR)/libmilhoja.a

MILHOJA_H = $(BUILDDIR)/Milhoja.h

include ./Makefile.configure
include $(SITE_MAKEFILE)

# Default shell commands
RM ?= /bin/rm

ifeq      ($(CXXCOMPNAME),gnu)
include ./gnu.mk
else ifeq ($(CXXCOMPNAME),intel)
include ./intel.mk
else
$(error $(CXXCOMPNAME) compiler not yet supported.)
endif

ifeq ($(DEBUG),true)
CXXFLAGS = -I$(INCDIR) -I$(BUILDDIR) $(CXXFLAGS_STD) $(CXXFLAGS_DEBUG) $(CXXFLAGS_AMREX)
else
CXXFLAGS = -I$(INCDIR) -I$(BUILDDIR) $(CXXFLAGS_STD) $(CXXFLAGS_PROD)  $(CXXFLAGS_AMREX)
endif

CPP_SRCS = \
	$(SRCDIR)/Milhoja_Logger.cpp \
	$(SRCDIR)/Milhoja_IntVect.cpp \
	$(SRCDIR)/Milhoja_RealVect.cpp \
	$(SRCDIR)/Milhoja_FArray4D.cpp \
	$(SRCDIR)/Milhoja_FArray3D.cpp \
	$(SRCDIR)/Milhoja_FArray2D.cpp \
	$(SRCDIR)/Milhoja_FArray1D.cpp \
	$(SRCDIR)/Milhoja_GridConfiguration.cpp \
	$(SRCDIR)/Milhoja_GridConfigurationAMReX.cpp \
	$(SRCDIR)/Milhoja_Grid.cpp \
	$(SRCDIR)/Milhoja_GridAmrex.cpp \
	$(SRCDIR)/Milhoja_Tile.cpp \
	$(SRCDIR)/Milhoja_TileAmrex.cpp \
	$(SRCDIR)/Milhoja_RuntimeElement.cpp \
	$(SRCDIR)/Milhoja_DataPacket.cpp \
	$(SRCDIR)/Milhoja_ThreadTeam.cpp \
	$(SRCDIR)/Milhoja_ThreadTeamIdle.cpp \
	$(SRCDIR)/Milhoja_ThreadTeamTerminating.cpp \
	$(SRCDIR)/Milhoja_ThreadTeamRunningOpen.cpp \
	$(SRCDIR)/Milhoja_ThreadTeamRunningClosed.cpp \
	$(SRCDIR)/Milhoja_ThreadTeamRunningNoMoreWork.cpp \
	$(SRCDIR)/Milhoja_MoverUnpacker.cpp \
	$(SRCDIR)/Milhoja_Runtime.cpp \
	$(SRCDIR)/Milhoja_RuntimeBackend.cpp

CPP_OBJS=$(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CPP_SRCS:.cpp=.o))

.PHONY: default all install clean spotless
default: $(TARGET)
all:     $(TARGET)
# TODO: We should only copy over those headers associated with the
#       specified backends.
install:
	$(RM) -r $(LIB_MILHOJA_PREFIX)
	mkdir -p $(LIB_MILHOJA_PREFIX)/include
	mkdir -p $(LIB_MILHOJA_PREFIX)/lib
	cp $(TARGET) $(LIB_MILHOJA_PREFIX)/lib
	cp $(BUILDDIR)/Milhoja.h $(LIB_MILHOJA_PREFIX)/include
	cp $(INCDIR)/*.h $(LIB_MILHOJA_PREFIX)/include
clean:
	$(RM) $(BUILDDIR)/*.o
	$(RM) $(BUILDDIR)/*.d
spotless:
	$(RM) -r $(BUILDDIR)

$(BUILDDIR):
	@mkdir $(BUILDDIR)

$(MILHOJA_H): | $(BUILDDIR)
	@./tools/write_library_header.py --dim $(NDIM) \
                                     --runtime $(RUNTIME_BACKEND) \
                                     --grid $(GRID_BACKEND) \
                                     --fps $(FLOATING_POINT_SYSTEM) \
                                     --offload $(COMPUTATION_OFFLOADING) \
                                     $(MILHOJA_H)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(MILHOJA_H) Makefile
	$(CXXCOMP) -c $(DEPFLAGS) $(CXXFLAGS) -o $@ $<

$(TARGET): $(CPP_OBJS) Makefile
	ar -rcs $@ $(CPP_OBJS)

