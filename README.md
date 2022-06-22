Requirements
============
* a python installation
* C++ compiler that supports C++11 and pthreads
* Fortran compiler that supports F2003
* MPI installation associated with the aforementioned C++ and Fortran compilers
* googletest built with same compilers that will be used to build tests
* Source-only installation of [nlohmann's C++ JSON library](https://github.com/nlohmann/json)
* AMReX [123]D libraries including Fortran interfaces built with same compilers/MPI implementation that will be used to build tests

Building Milhoja as a library
=================================

This repository contains a library build system for constructing Milhoja as static libraries.  This is the official Milhoja build system and users interface with it primarily through the creation of site-specific Makefiles and the `configure.py` script found at the root of the repository.

Some examples of site-specific Makefiles can be found in the `sites` folder.  The `gce` files are in use for CPU-only CI testing and the `summit` folder is used for general development and testing.  Users are free to construct their own Makefiles and these need not be included in this repository.

Each Milhoja library is built to use
* a single dimension for the domain of problems to be solved with Milhoja (e.g., 1, 2, 3),
* a single runtime backend (e.g., CUDA),
* a single Grid backend (e.g., AMReX), and
* a single computational offloading model (e.g., OpenACC).

In addition, libraries can be built in debug or production mode, where the Makefile in use defines what those two modes entail.  Please run `configure.py -h` for more information on using that tool.

The configure script creates a file called `Makefile.configure`.  With this in place, the desired library can be constructed by running `make` from the root of the repository and installed in the specified location by running `make install`.

Building Tests
==============
The test portion of this repository is considered to be external to Milhoja and has its own dedicated build system.  The tests and their infrastructure are potentially subject to more frequent and substantial changes than those of Milhoja.

Unlike libraries, tests can only be built at present using Makefiles located in the `sites` folder.  It is, of course, recommended to build libraries and tests with the same compilers/MPI implementation and site-specific Makefile.

Tests are setup using the `setup.py` script in the root of the repository with commands such as

- setup.py Grid/{general,gcfill,multiple} -d {1,2,3} -s gce -l ~/local/Milhoja_{1,2,3}D_sandybridge_intel_mpich_debug -p grid\_{1,2,3}d.json
- setup.py ThreadTeam -d 2 -s gce -l ~/local/Milhoja_2D_sandybridge_intel_mpich_debug -p threadteam.json
- setup.py Runtime/null -d 2 -s gce -l ~/local/Milhoja_2D_sandybridge_intel_mpich_debug -p runtime.json
- setup.py Runtime/cpu -d {2,3} -s gce -l ~/local/Milhoja_{2,3}D_sandybridge_intel_mpich_debug -p runtime.json
- setup.py Runtime/gpu -d {2,3} -s summit -l ~/local/Milhoja_{2,3}D_nvhpc_cuda_openacc -p runtime.json
- setup.py Sedov/mpi -d {2,3} -s gce -l ~/local/Milhoja_{2,3}D_sandybridge_intel_mpich_debug -p sedov\_{2,3}D\_cartesian\_cpu.par 
- setup.py Sedov/cpu -d {2,3} -s gce -l ~/local/Milhoja_{2,3}D_sandybridge_intel_mpich_debug -p sedov\_{2,3}D\_cartesian\_cpu.par 
- setup.py Sedov/gpu/variant{1,2,3} -d {2,3} -s summit -l ~/local/Milhoja_{2,3}D_nvhpc_cuda_openacc -p sedov\_{2,3}D\_cartesian\_gpu.par

Note that the name of a test is the path of the test relative to the `test` folder and that tests can use runtime parameters encoded in JSON-format files.  Please run `setup.py -h` for more information on using that tool.

The setup tool creates and populates a build folder of the specified name in the root of the repository.  The test is built by running `make` from within that folder.

