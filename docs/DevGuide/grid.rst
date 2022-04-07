The Grid Abstract Interface
===========================

Calling code might have its own set of requirements that dictate what values, if
any, certain variables should have for dimensions above MILHOJA_NDIM.  Therefore,
this library should __not__ impose on the calling code what those values should
be.  Therefore, the preferred interface design should either
* not return any values above MILHOJA_NDIM or
* not set values above MILHOJA_NDIM so that calling code can set such values
  before calling the library without fear of overwriting.
Implementations should avoid storing any data above MILHOJA_NDIM in the name of
simplicity, cleanliness, and defensive programming.  If this is unavoidable,
then garbage default values that should lead to failures or obviously wrong
values should be used.  These same rules should be propagated through to the
Fortran/C++ interoperability layer.

Some of the affected variables are
* [yz]Min/Max
* n[yz]b
* nBlocks[YZ]
* deltas Y & Z
* Y & Z cell/face/edge/node indices
* Y & Z cell volumes/face areas/edge lengths

Similarly, the configuration of the Grid shall ignore all variables related to
dimensions above MILHOJA_NDIM.  Therefore, correct functionality shall arise
even if calling code does not set such values.

Implementations shall not include guardcells for dimensions above MILHOJA_NDIM
and the number of cells along a dimension above MILHOJA_NDIM shall be constant
for all levels.

TODO: Add in table that defines index sets in each Grid backend, in the Milhoja
C++ interface, in the C/C++ interoperability layer, and in the Fortran
interface.  Obviously it would be nice to have the data collected in one place
so that people can see the big picture quickly.  However, I do like the idea
that each routine specify explicitly what it deals with as part of its
interface/contract.  Therefore, any docs here should state that the inline docs
supersede what is written here, which is just a general goal for improving
consistency throughout code layers.

Grid use & lifetime sequence
****************************

1. The application is built with its selection of Grid backend known and fixed
at compile time.

2. The application loads its runtime parameters, if any, using its own
facilities.

3. The application instantiates the GridConfiguration singleton.

   * Behind the scenes, the concrete derived class associated with the desired
     backend is instantiated automatically.

4. The application sets all configuration values stored in the GridConfiguration
instance using fixed and runtime parameter values.

5. The application calls the GridConfiguration instance's load member function.

   * Ideally, the load function shall validate all given configuration values.
   * For AMReX, this loads the supplied configuration values into AMReX using
     its ParmParse facility.
   * For other backends, this might do nothing more than validation.

6. The application instantiates the Grid singleton.

7. Behind the scenes, the desired backend is instantiated automatically.

   * The backend's constructor is executed automatically.
   * The backend's constructor can access and store configuration values stored
     in the GridConfiguration singleton.  It might do so if it owns the data
     (e.g., NGUARD, N[XYZ]B, nCcVars in the case of AMReX).
   * The backend's constructor clears the GridConfiguration singleton.

8. The application calls the Grid singleton's initDomain routine and specifies
that it should use a particular ThreadTeam with a given set of parameter values
to set the problem's initial conditions into the Grid's data structures.

9. At the end of the simulation, the application calls the Grid singleton's
destroyDomain.

10. The application calls the Grid singleton's finalize static member function.

11. The compiler dictates when the Grid and GridConfiguration singletons are
destroyed.

AMReX Backend
-------------

The ownership of Grid unit configuration values is split between AMReX and the
GridAmrex class.  This division of ownership is motivated by the goal to only
have configuration values stored in one location so that synchronization of
values across multiple variables is not an issue.  To this end, if AMReX is
configured with a value and has a getter for accessing the value, then AMReX
owns the value.  GridAmrex generally owns values that are used by the AmrCore
member functions that it overrides to construct and manage MultiFabs.

The caching of configuration values owned by AMReX in GridAmrex is considered
preoptimization presently and has therefore been avoided.

==============   =====    =========
Configuration    AMReX    GridAmrex
==============   =====    =========
(xMin, xMax)     X
(yMin, yMax)     X
(zMin, zMax)     X
maxFinestLevel   X
nxb                       X
nyb                       X
nzb                       X
nGuard                    X
nCcVars                   X
nBlocksX                  X
nBlocksY                  X
nBlocksZ                  X
==============   =====    =========

