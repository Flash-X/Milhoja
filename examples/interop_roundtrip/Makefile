MILHOJA_PREFIX ?= ../../2D-hostmem
PYTHON ?= python3
MILHOJA_PYTHONPATH ?= ../../tools/milhoja_pypkg/src
LAYOUT ?= variable-last
EXECUTION ?= hostmem
GPU_ARCH ?= gfx1030
HIP_LIB_DIR ?=

ifneq ($(strip $(HIP_LIB_DIR)),)
HIP_LIBRARY_PATH_FLAGS := -L$(HIP_LIB_DIR) -Wl,-rpath,$(HIP_LIB_DIR)
else
HIP_LIBRARY_PATH_FLAGS :=
endif

ifeq ($(LAYOUT),variable-last)
LAYOUT_FLAGS :=
else ifeq ($(LAYOUT),variable-first)
LAYOUT_FLAGS := -DROUNDTRIP_VARIABLE_FIRST
else
$(error Unsupported LAYOUT '$(LAYOUT)'; use variable-last or variable-first)
endif

GENERATED := generated/$(EXECUTION)/$(LAYOUT)
BUILD := build/$(EXECUTION)/$(LAYOUT)

ifeq ($(EXECUTION),hostmem)
FC = gfortran
CXX = c++
MPICXX = mpicxx
LINKER = mpifort
CXX_RUNTIME ?= -lc++
MIXED_LINK_FLAGS :=
RUNNER ?=
OFFLOAD_FLAGS :=
CXX_OFFLOAD_FLAGS :=
FFLAGS := -cpp -std=f2008 -Wall -Wextra -fcheck=all \
          $(LAYOUT_FLAGS) \
          -I$(MILHOJA_PREFIX)/include -I$(BUILD) -J$(BUILD)
else ifeq ($(EXECUTION),cuda-openacc)
FC = mpifort
CXX = mpicxx
MPICXX = mpicxx
LINKER = mpicxx
CXX_RUNTIME ?=
MIXED_LINK_FLAGS := -fortranlibs -cudalib
RUNNER ?= env OMPI_MCA_coll_ucc_enable=0 OMPI_MCA_pml=ob1 \
          OMPI_MCA_btl=self UCX_MODULES=^cuda mpirun -np 1
OFFLOAD_FLAGS ?= -acc=gpu -gpu=ccnative,cuda13.1,ptxinfo -Minfo=accel
CXX_OFFLOAD_FLAGS := $(OFFLOAD_FLAGS)
FFLAGS := -Mpreprocess -Mrecursive -i4 -r8 -traceback \
          $(OFFLOAD_FLAGS) $(LAYOUT_FLAGS) \
          -I$(MILHOJA_PREFIX)/include -I$(BUILD) -module $(BUILD)
else ifeq ($(EXECUTION),hip-openmp)
FC = mpifort
CXX = mpicxx
MPICXX = mpicxx
LINKER = mpifort
CXX_RUNTIME ?= -lstdc++
MIXED_LINK_FLAGS := $(HIP_LIBRARY_PATH_FLAGS) -lamdhip64
RUNNER ?= timeout 90s env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
          OMP_DEFAULT_DEVICE=0 OMP_TARGET_OFFLOAD=MANDATORY mpirun -np 1
OFFLOAD_FLAGS ?= -fopenmp -foffload=amdgcn-amdhsa \
                  -foffload-options=amdgcn-amdhsa=-march=$(GPU_ARCH)
CXX_OFFLOAD_FLAGS := -D__HIP_PLATFORM_AMD__
FFLAGS := -O2 -cpp -std=f2008 -Wall -Wextra \
          $(OFFLOAD_FLAGS) $(LAYOUT_FLAGS) \
          -I$(MILHOJA_PREFIX)/include -I$(BUILD) -J$(BUILD)
else
$(error Unsupported EXECUTION '$(EXECUTION)'; use hostmem, cuda-openacc, or hip-openmp)
endif

MPI_CXXFLAGS = $(shell $(MPICXX) --showme:compile)
TARGET := $(BUILD)/interop_roundtrip
STAMP := $(GENERATED)/.generated
GENERATED_FILES := \
  $(GENERATED)/DataPacket_roundtrip_tf.cxx \
  $(GENERATED)/DataPacket_roundtrip_tf.h \
  $(GENERATED)/roundtrip_tf.cxx \
  $(GENERATED)/roundtrip_tf_C2F.F90 \
  $(GENERATED)/roundtrip_tf_mod.F90

CXXFLAGS := -std=c++17 -Wall -Wextra $(CXX_OFFLOAD_FLAGS) $(LAYOUT_FLAGS) \
            -I$(MILHOJA_PREFIX)/include -I$(GENERATED)

OBJECTS := \
  $(BUILD)/roundtrip_kernel_mod.o \
  $(BUILD)/roundtrip_tf_mod.o \
  $(BUILD)/roundtrip_tf_C2F.o \
  $(BUILD)/DataPacket_roundtrip_tf.o \
  $(BUILD)/roundtrip_tf.o \
  $(BUILD)/roundtrip_main.o

.PHONY: all generate generate-both regenerate regenerate-both \
        verify-generation run run-both clean

all: generate $(TARGET)

generate: $(STAMP)

generate-both:
	$(MAKE) generate EXECUTION=$(EXECUTION) LAYOUT=variable-last
	$(MAKE) generate EXECUTION=$(EXECUTION) LAYOUT=variable-first

$(STAMP): generate_roundtrip.py
	PYTHONPATH=$(MILHOJA_PYTHONPATH):$$PYTHONPATH \
	  $(PYTHON) generate_roundtrip.py $(MILHOJA_PREFIX) $(GENERATED) \
	  --execution $(EXECUTION) --layout $(LAYOUT) --overwrite
	touch $(STAMP)

regenerate:
	PYTHONPATH=$(MILHOJA_PYTHONPATH):$$PYTHONPATH \
	  $(PYTHON) generate_roundtrip.py $(MILHOJA_PREFIX) $(GENERATED) \
	  --execution $(EXECUTION) --layout $(LAYOUT) --overwrite
	touch $(STAMP)

regenerate-both:
	$(MAKE) regenerate EXECUTION=$(EXECUTION) LAYOUT=variable-last
	$(MAKE) regenerate EXECUTION=$(EXECUTION) LAYOUT=variable-first

verify-generation:
	PYTHONPATH=$(MILHOJA_PYTHONPATH):$$PYTHONPATH \
	  $(PYTHON) verify_generation.py $(MILHOJA_PREFIX)

$(GENERATED_FILES): $(STAMP)

run: all
	$(RUNNER) $(TARGET)

run-both:
	$(MAKE) run EXECUTION=$(EXECUTION) LAYOUT=variable-last
	$(MAKE) run EXECUTION=$(EXECUTION) LAYOUT=variable-first

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/roundtrip_kernel_mod.o: roundtrip_kernel_mod.F90 | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/roundtrip_tf_mod.o: $(GENERATED)/roundtrip_tf_mod.F90 $(BUILD)/roundtrip_kernel_mod.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/roundtrip_tf_C2F.o: $(GENERATED)/roundtrip_tf_C2F.F90 $(BUILD)/roundtrip_tf_mod.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/DataPacket_roundtrip_tf.o: $(GENERATED)/DataPacket_roundtrip_tf.cxx | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/roundtrip_tf.o: $(GENERATED)/roundtrip_tf.cxx | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/roundtrip_main.o: roundtrip_main.cpp $(GENERATED)/DataPacket_roundtrip_tf.h | $(BUILD)
	$(CXX) $(MPI_CXXFLAGS) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(LINKER) $(OBJECTS) $(MILHOJA_PREFIX)/lib/libmilhoja.a \
	  $(OFFLOAD_FLAGS) $(MIXED_LINK_FLAGS) $(CXX_RUNTIME) -lpthread -o $@

clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.mod $(TARGET)
