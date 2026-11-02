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

This writes independent sources beneath `generated/variable-last` and
`generated/variable-first`. To force regeneration after a generator change,
run `make regenerate-both` with the same prefix.

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
Milhoja Step 5D HOSTMEM variable-last round trip passed
Milhoja Step 5D HOSTMEM variable-first round trip passed
```

The generated and object files are isolated under layout-specific directories.
On GNU/Linux, add `CXX_RUNTIME=-lstdc++` to the `make` commands.
