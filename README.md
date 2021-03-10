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
