HIPCC ?= hipcc
# GNU make defines FC=f77 as a built-in default. Use a normal assignment so
# this example selects gfortran unless FC is explicitly set on the command
# line or by a higher-priority make override.
FC = gfortran

GPU_ARCH ?= gfx1030
OFFLOAD_TARGET ?= amdgcn-amdhsa
HIP_LIB_DIR ?=

BUILD := build
TARGET := $(BUILD)/hip_omp_interop

OFFLOAD_FLAGS := -fopenmp -foffload=$(OFFLOAD_TARGET) \
                 -foffload-options=$(OFFLOAD_TARGET)=-march=$(GPU_ARCH)
FFLAGS := -O2 -std=f2018 -Wall -Wextra -J$(BUILD) -I$(BUILD) \
          $(OFFLOAD_FLAGS)
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra
ifneq ($(strip $(HIP_LIB_DIR)),)
HIP_LIBRARY_PATH_FLAGS := -L$(HIP_LIB_DIR) -Wl,-rpath,$(HIP_LIB_DIR)
endif

LINK_FLAGS := $(OFFLOAD_FLAGS) $(HIP_LIBRARY_PATH_FLAGS) \
              -lamdhip64 -lstdc++

OBJECTS := \
  $(BUILD)/hip_omp_kernel.o \
  $(BUILD)/hip_omp_main.o

.PHONY: all check-environment run clean

all: $(TARGET)

check-environment:
	@command -v $(HIPCC)
	@command -v $(FC)
	@$(HIPCC) --version | head -n 1
	@$(FC) --version | head -n 1
	@echo "GPU_ARCH=$(GPU_ARCH)"
	@echo "OFFLOAD_TARGET=$(OFFLOAD_TARGET)"
	@echo "HIP_LIB_DIR=$(if $(HIP_LIB_DIR),$(HIP_LIB_DIR),<system linker search>)"

$(BUILD):
	mkdir -p $@

$(BUILD)/hip_omp_kernel.o: hip_omp_kernel.F90 | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/hip_omp_main.o: hip_omp_main.cpp | $(BUILD)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(FC) $(OBJECTS) $(LINK_FLAGS) -o $@

run: check-environment $(TARGET)
	ROCR_VISIBLE_DEVICES=0 \
	HIP_VISIBLE_DEVICES=0 \
	OMP_DEFAULT_DEVICE=0 \
	OMP_TARGET_OFFLOAD=MANDATORY \
	$(TARGET)

clean:
	rm -rf $(BUILD)

