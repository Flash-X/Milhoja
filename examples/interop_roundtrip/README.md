# Milhoja interoperability round trip

## Step 5B minimal example

This Milhoja-only example validates the compiler and runtime interface without
Spark or RecipeTools. It packs two `3 x 2` interior tiles with one guard layer
and four components, calls a generated C++ -> C -> Fortran TaskFunction, and
unpacks the transformed in/out field.

Every input encodes its tile, component, and spatial index. The Fortran kernel
also consumes an external scale and tile lower-bound metadata, and produces a
per-tile scratch checksum. The driver checks every returned field value and
both checksums.

## Step 5C dual-layout generation

Generate persistent source sets for both layouts with:

```sh
make generate-both MILHOJA_PREFIX=/path/to/Milhoja/2D-hostmem
```

This writes independent sources beneath `generated/hostmem/variable-last` and
`generated/hostmem/variable-first`. To force regeneration after a generator
change, run `make regenerate-both` with the same prefix.

Generate and inspect both layouts without compiling:

```sh
make verify-generation MILHOJA_PREFIX=/path/to/Milhoja/2D-hostmem
```

The verifier generates each layout twice in temporary directories. It checks
the complete file set, TaskFunction and DataPacket metadata, in/out and scratch
descriptions, exact C-to-Fortran shapes, variable-first transposition,
variable-last contiguous copies, absence of OpenACC code, and byte-for-byte
determinism.

## Step 5D HOSTMEM execution

Build and run both HOSTMEM layouts with one command:

```sh
make run-both MILHOJA_PREFIX=/path/to/Milhoja/2D-hostmem
```

The driver verifies the full packet lifecycle in addition to numerical values:

- `pack` acquires the only configured stream;
- Tile storage remains unchanged while computation modifies packet memory;
- the return-transfer callback completes and invokes `unpack`;
- `unpack` releases the stream;
- every field value and both per-tile scratch checksums match the analytic
  result; and
- backend and logger finalization complete after packet destruction.

The layouts can also be run separately:

```sh
make run LAYOUT=variable-last \
  MILHOJA_PREFIX=/path/to/Milhoja/2D-hostmem

make run LAYOUT=variable-first \
  MILHOJA_PREFIX=/path/to/Milhoja/2D-hostmem
```

The combined run must end with both messages:

```text
Milhoja interoperability HOSTMEM variable-last round trip passed
Milhoja interoperability HOSTMEM variable-first round trip passed
```

The generated and object files are isolated under layout-specific directories.
On GNU/Linux, add `CXX_RUNTIME=-lstdc++` to the `make` commands.

## Step 5E NVIDIA CUDA/OpenACC execution

Step 5E uses the same kernel and lifecycle driver with a CUDA-runtime Milhoja
installation. The generator emits a GPU TaskFunction using OpenACC, including
device pointers, an asynchronous queue, a gang loop over tiles, and the
generated C++/Fortran interface. The kernel runs its field transformation and
checksum reduction on the GPU. The checksum is also written to the logical
first field element so the driver can validate it after the normal device-to-
host transfer without dereferencing device scratch memory.

On the NVIDIA platform, first activate the NVHPC and HPC-X environment used to
configure Milhoja. Then, from this directory, run the source-only checks:

```sh
make verify-generation \
  MILHOJA_PREFIX=/home/adubey/ORCHA/Milhoja/2D-cuda
```

Generate, compile, and run both data layouts on the GPU with:

```sh
make run-both EXECUTION=cuda-openacc \
  MILHOJA_PREFIX=/home/adubey/ORCHA/Milhoja/2D-cuda
```

The default GPU flags target the visible GPU with NVHPC 26.3 and its CUDA 13.1
toolkit:

```text
-acc=gpu -gpu=ccnative,cuda13.1,ptxinfo -Minfo=accel
```

The Fortran compilation also uses `-i4 -r8`. Milhoja's OpenACC generators
intentionally emit default `integer` and `real` declarations in their device
wrappers, so these flags bind those declarations to the configured Milhoja
32-bit integer and 64-bit real types.

The CUDA executable is linked with the MPI C++ wrapper because the driver owns
the program's C++ `main`. The `-fortranlibs` option supplies the NVFORTRAN
runtime used by the generated Fortran objects, and `-cudalib` supplies the CUDA
runtime required by Milhoja's CUDA backend.

They can be replaced without editing the Makefile, for example:

```sh
make run-both EXECUTION=cuda-openacc \
  MILHOJA_PREFIX=/home/adubey/ORCHA/Milhoja/2D-cuda \
  OFFLOAD_FLAGS='-acc=gpu -gpu=ccnative,cuda13.1,ptxinfo -Minfo=accel'
```

A successful combined run ends with:

```text
Milhoja interoperability CUDA/OpenACC variable-last round trip passed
Milhoja interoperability CUDA/OpenACC variable-first round trip passed
```

Generated and compiled CUDA/OpenACC artifacts are kept under
`generated/cuda-openacc/<layout>` and `build/cuda-openacc/<layout>`, separate
from the HOSTMEM artifacts.
