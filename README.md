Requirements
============
* MPI installation
* C++ compiler that supports C++11 and pthreads
* Fortran compiler that supports F2003
* googletest built with same compilers that will be used to build tests
* AMReX [123]D libraries including Fortran interfaces built with same compilers that will be used to build tests

Building Tests
==============
If the user has a site directory configured as specified in the Build System Requirements document (build-system-reqs.txt), they should be able to run a number of unit tests, as follows:

- python setup.py -t Grid -d {1,2,3} -p grid\_{1,2,3}D.par
- python setup.py -t CudaBackend -d 2 -p par\_cudabackend.h
- python setup.py -t Runtime/null -d 2 -p par\_runtime.h
- python setup.py -t Runtime/cpu -d 2 -p par\_runtime.h
- python setup.py -t Runtime/cuda -d 2 -p par\_runtime.h
- python setup.py -t ThreadTeam -d 2 -p par\_threadteam.h
- python setup.py -t Sedov/mpi -d {2,3} -p sedov\_{2,3}D\_cartesian\_cpu.par 
- python setup.py -t Sedov/cpu -d {2,3} -p sedov\_{2,3}D\_cartesian\_cpu.par 
- python setup.py -t Sedov/gpu -d {2,3} -p sedov\_{2,3}D\_cartesian\_gpu.par


Building the Runtime as a library
=================================

By default (e.g. python setup.py Grid -d 2 -p grid\_2d.par), the Runtime is built as a library as an intermediate step, and then linked as a static library into the test executable. This need not be the case in the future.

In order to build the Runtime as a standalone library that can be linked from multiple tests, or from an external application like FLASH, use the special keyword `library` in place of a test name. (e.g. python setup.py library -d 2 --par grid\_2d.par --prefix runtime\_2d\_AMReX\_Cuda) The `prefix` command line option specifies a subdirectory (of the repo root dir) where the user can install library with `make install`.

To link a prebuilt library into a test, use the command line field `--library` to specify a path. This reduces the total amount of compilation time if running multiple tests with the same Runtime configuration.

TODO: More investagation still needs to be done to determine the behavior if a test links a library configured with different runtime or setup parameters.
