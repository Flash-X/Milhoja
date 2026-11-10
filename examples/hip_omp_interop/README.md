# HIP allocation to GNU Fortran OpenMP-target interoperability

This standalone prerequisite test asks two narrow questions before a Milhoja
HIP backend is implemented: can a GNU Fortran OpenMP target region directly
use device memory allocated by the platform HIP runtime, and does that path
preserve both data-layout conventions required by the generated tasks?

The test deliberately has no Milhoja, RecipeTools, or MPI dependency. The C++
driver constructs a logical `5 x 3 x 2` field with four variables. It packs
that field independently in variable-last and variable-first order, allocates
each physical field with HIP, passes its HIP device pointer through a `bind(C)`
interface, and invokes the matching GNU Fortran rank-four OpenMP target loop
with `is_device_ptr`. Non-unit, unequal extents and index-dependent values make
permuted dimensions or incorrect strides observable. The driver synchronizes
HIP, copies each field back, and verifies every logical value. Each Fortran
region must also report that it ran on a non-host OpenMP device.

The OpenMP kernel distributes only its outer, layout-contiguous loop. The
inner loops remain serial within each worker. This deliberately follows the
smallest GPU code-generation pattern already established by the scalar Stage A
probe; multidimensional collapsed-loop optimization is outside this layout
interoperability prerequisite.

The OpenMP specification makes support for device pointers created outside
OpenMP implementation-defined. Passing this test is therefore a prerequisite
for using HIP-owned Milhoja DataPacket storage directly from generated GNU
Fortran OpenMP kernels.

## Required environment

On the RX 6800 XT node, load the architecture-specific ROCm and GNU-compatible
MPI modules. MPI is not used by this test, but loading the intended platform
environment now makes subsequent Milhoja work reproducible.

```sh
module purge
module load rocm/gfx10
module load mpi/openmpi-x86_64
```

The following Rocky Linux packages must also be installed for GCC AMDGCN
offload:

```text
gcc-offload-amdgcn
libgomp-offload-amdgcn
```

## Build and run

From `examples/hip_omp_interop`, inspect the selected tools and paths:

```sh
make check-environment
```

Build and run with mandatory GPU offload. On the Rocky Linux installation the
HIP runtime is normally in the system linker's standard search path, so do not
pass the module's `ROCM_LIB`; that directory can contain architecture-specific
support files rather than `libamdhip64`.

```sh
make run
```

The defaults select `gfx1030`, the `amdgcn-amdhsa` GCC offload target, and the
HIP library installed by the `rocm/gfx10` module. They can be overridden
without editing the Makefile, for example:

```sh
make run \
  GPU_ARCH=gfx1030 \
  HIP_LIB_DIR=/nonstandard/path/containing/libamdhip64
```

A successful run ends with output similar to:

```text
variable-last layout passed
variable-first layout passed
HIP device: AMD Radeon RX 6800 XT
HIP allocation -> GNU Fortran OpenMP target -> HIP copy-back passed for both layouts
```

`OMP_TARGET_OFFLOAD=MANDATORY` is set by the `run` target, so an unavailable
offload device is a hard error rather than a silent host fallback. The test is
synchronous by design. Stream sharing and asynchronous ordering belong to the
later Milhoja HIP-backend stage and are not claimed here.

If compilation succeeds but execution rejects the external device pointer,
the next isolated experiment will use `omp_target_associate_ptr`; do not add a
Milhoja workaround before that alternative is tested.

