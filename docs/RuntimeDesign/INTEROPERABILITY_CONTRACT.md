# Milhoja compiler/backend interoperability contract

This document is the Step 5A audit and design contract for a minimal
DataPacket round trip. It describes the interfaces that exist now and fixes
the boundary for Step 5B. It does not add a TaskFunction, a kernel, or a new
backend.

## Independent configuration axes

Milhoja exposes three choices that must remain independent:

1. **Runtime backend** selects memory, transfer, and stream operations. The
   configured implementations are `HOSTMEM`, `CUDA`, and `None`.
2. **Computation offloading** selects the kernel execution model. The configured
   implementations are `OpenACC` and `None`.
3. **Language interoperability** selects the generated C++ TaskFunction,
   C-compatible entry point, and Fortran implementation used for a Fortran
   DataPacket TaskFunction.

The build records the first two choices separately as `RUNTIME_BACKEND` and
`COMPUTATION_OFFLOADING`. Generated code must not infer either choice merely
from the use of a DataPacket.

`data_layout` is a fourth, orthogonal generation choice. Its supported values
are `variable-last` and `variable-first`; the batched tile axis remains last.

## Audited configuration matrix

| Runtime | Offload | Fortran DataPacket generation | Step 5 status |
| --- | --- | --- | --- |
| HOSTMEM | None | Present: CPU/DataPacket dispatch emits host Fortran, C-to-Fortran, C++ wrapper, and packet module | Locally testable reference configuration |
| HOSTMEM | OpenACC | Source and fake-stream paths exist | Emulation/compile configuration only; not proof of device memory or device execution |
| CUDA | OpenACC | Present: established OpenACC Fortran DataPacket path | Intended GPU validation configuration; requires a CUDA/OpenACC target |
| CUDA | None | No Fortran DataPacket TaskFunction path is selected by the current dispatcher | Unsupported for Step 5 |
| None | None | No usable DataPacket transfer or stream backend | Unsupported for a round trip |

References to `MILHOJA_CUDA_OFFLOADING` exist in low-level headers, but CUDA
kernel offloading is not a selectable computation model in the current
configuration tool. It is therefore not a supported Step 5 configuration.

## Existing interface chain

The round trip must use the existing chain without bypassing a layer:

```text
Tile storage
  -> generated DataPacket::pack
  -> RuntimeBackend::initiateHostToGpuTransfer
  -> generated C++ TaskFunction wrapper
  -> bind(C) C-to-Fortran routine
  -> generated Fortran TaskFunction and test computation
  -> RuntimeBackend::initiateGpuToHostTransfer
  -> callback -> generated DataPacket::unpack
  -> Tile storage
```

The HOSTMEM backend implements the two transfers with synchronous `memcpy` and
invokes the return callback synchronously. The CUDA backend uses
`cudaMemcpyAsync` and `cudaLaunchHostFunc` on the packet's CUDA stream. The
same lifecycle must be followed in both cases even though HOSTMEM completes
each operation before returning.

## C++/C/Fortran ABI contract

- Generated C++ passes the packet handle, queue identifiers, tile count, and
  device-side member pointers through an `extern "C"` function.
- The C-to-Fortran routine uses `bind(c)`, receives raw addresses as `C_PTR`,
  and establishes Fortran pointers with `C_F_POINTER` and explicit shapes.
- The current Milhoja build supports IEEE double precision only. C++
  `milhoja::Real`, C-compatible real arguments, and Fortran `MILHOJA_REAL`
  must therefore agree with double precision.
- C-compatible integer scalars use Milhoja's configured integer kind. No test
  may assume that a Fortran default integer is ABI-compatible without the
  generated interface.
- `variable-last` grid shapes are `(i,j,k,variable,tile)`.
- `variable-first` grid shapes are `(variable,i,j,k,tile)`.
- Metadata shapes and scalar values are independent of grid-data layout.
- HOSTMEM wrappers pass queue identifier zero and emit no OpenACC dependency.
- OpenACC wrappers obtain the packet's asynchronous queue. The CUDA stream
  manager maps that queue to the CUDA stream used for transfers.

## Ownership and lifetime contract

1. RuntimeBackend must be initialized before constructing or packing a packet.
2. `DataPacket::addTile` consumes the caller's moved `shared_ptr` and retains
   the tile until it is popped or the packet is destroyed.
3. Constructor/external values are copied into the concrete packet.
4. `pack` requires at least one tile, obtains the paired host/device memory
   blocks from RuntimeBackend, fills host packet storage, and acquires the
   packet's main stream.
5. A packet owns its acquired memory and stream exclusively. A second `pack`
   while those resources remain assigned is an error.
6. Computation may use packet device pointers only after the host-to-device
   transfer has been initiated in the packet's stream/queue order.
7. Device-to-host transfer must precede `unpack`. The return callback performs
   unpack only after returned data is available.
8. `unpack` releases the main stream before copying returned values into Tile
   storage. Extra OpenACC queues, if requested, must be released by the
   TaskFunction first.
9. Packet destruction releases any remaining stream and paired memory. Normal
   completion must release the stream before RuntimeBackend finalization.
10. RuntimeBackend and Logger must be finalized after packet work and before
    process termination.

## Synchronization contract

- The packet's main stream orders input transfer, computation, output transfer,
  and callback.
- A TaskFunction must not launch computation into an unrelated default stream.
- HOSTMEM's synchronous implementation does not relax the ordering contract;
  code validated only because operations happen immediately is invalid.
- Returned Tile data must not be inspected before the callback has completed.
- A Step 5 GPU result is valid only if the kernel and both transfers use the
  intended shared stream/queue and the callback observes completed output.

## Step 5B test boundary

Step 5B will add a new Milhoja-only example, independent of Spark and
RecipeTools, with these fixed properties:

- two tiles;
- asymmetric two-dimensional interior extents `3 x 2`;
- four components and nontrivial guard-cell bounds;
- a DataPacket in/out field, one external real scale, and one diagnostic
  scratch value;
- input values that uniquely encode tile, component, and spatial indices;
- a Fortran computation with an analytically reproducible result;
- generation and execution for both `variable-last` and `variable-first`;
- HOSTMEM as the first runtime configuration.

The test must detect component transposition, spatial stride, tile offset,
lower-bound, pointer-shape, ownership, stream, and synchronization errors. It
must compare logical values, not raw byte sequences across different layouts.

Step 5B may add example and test sources but must not add a new runtime backend
or change Spark arithmetic. Device compilation and execution remain later
Step 5 stages.

## Step 5A acceptance criteria

Step 5A is complete when:

- this matrix agrees with configuration and generator dispatch;
- transfer and callback semantics agree with both concrete backends;
- ABI, ownership, and synchronization rules agree with current interfaces;
- unsupported combinations are identified explicitly; and
- the adjacent source-audit verifier passes.
