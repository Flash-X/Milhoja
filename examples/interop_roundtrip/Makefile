MILHOJA_PREFIX ?= ../../2D-hostmem
PYTHON ?= python3
LAYOUT ?= variable-last
FC = gfortran
CXX = c++
MPICXX = mpicxx
LINKER = mpifort
CXX_RUNTIME ?= -lc++
MPI_CXXFLAGS = $(shell $(MPICXX) --showme:compile)

ifeq ($(LAYOUT),variable-last)
LAYOUT_FLAGS :=
else ifeq ($(LAYOUT),variable-first)
LAYOUT_FLAGS := -DROUNDTRIP_VARIABLE_FIRST
else
$(error Unsupported LAYOUT '$(LAYOUT)'; use variable-last or variable-first)
endif

GENERATED := generated/$(LAYOUT)
BUILD := build/$(LAYOUT)
TARGET := $(BUILD)/interop_roundtrip
STAMP := $(GENERATED)/.generated
GENERATED_FILES := \
  $(GENERATED)/DataPacket_roundtrip_tf.cxx \
  $(GENERATED)/DataPacket_roundtrip_tf.h \
  $(GENERATED)/roundtrip_tf.cxx \
  $(GENERATED)/roundtrip_tf_C2F.F90 \
  $(GENERATED)/roundtrip_tf_mod.F90

FFLAGS := -cpp -std=f2008 -Wall -Wextra -fcheck=all \
          $(LAYOUT_FLAGS) \
          -I$(MILHOJA_PREFIX)/include -I$(BUILD) -J$(BUILD)
CXXFLAGS := -std=c++17 -Wall -Wextra $(LAYOUT_FLAGS) \
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
	$(MAKE) generate LAYOUT=variable-last
	$(MAKE) generate LAYOUT=variable-first

$(STAMP): generate_roundtrip.py
	$(PYTHON) generate_roundtrip.py $(MILHOJA_PREFIX) $(GENERATED) \
	  --layout $(LAYOUT) --overwrite
	touch $(STAMP)

regenerate:
	$(PYTHON) generate_roundtrip.py $(MILHOJA_PREFIX) $(GENERATED) \
	  --layout $(LAYOUT) --overwrite
	touch $(STAMP)

regenerate-both:
	$(MAKE) regenerate LAYOUT=variable-last
	$(MAKE) regenerate LAYOUT=variable-first

verify-generation:
	$(PYTHON) verify_generation.py $(MILHOJA_PREFIX)

$(GENERATED_FILES): $(STAMP)

run: all
	$(TARGET)

run-both:
	$(MAKE) run LAYOUT=variable-last
	$(MAKE) run LAYOUT=variable-first

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
	  $(CXX_RUNTIME) -lpthread -o $@

clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.mod $(TARGET)
