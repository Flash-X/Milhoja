Requirements
============
The statements made in this section are intended to be informal, high-level
requirements.  Specifically, they should be correct for all orchestration
runtime designs and ideally require little change if our design changes
significantly.  In other words, these statements should not be informed by
implementation ideas or details.

System
******

The design philosophy for configuring the various components used in the system
is to
* avoid as much as possible having multiple copies of any single value,
* avoid the notion of runtime parameters,
* keep the configuration systems as minimal and simple as possible, and
* grow the complexity of configuration systems only as needed as new backends or
  features are added.

In particular, calling code is asked to simply pass configuration values to this
system.  Therefore, it is irrelevant from the point of view of this library if
these values were fixed at compile time or loaded/determined dynamically.

1. A build system shall be setup that is dedicated solely to building the
Milhoja library.  It shall be similar to traditional build systems in the sense
that it will build site- and software-stack specific versions of the library
without users manually creating a site-specific Makefile.

2. A build system separate from the library build system shall be designed.
It's interface shall aid in not only building, but also configuring each test.
The design of this system is very much influenced by the Flash-X setup tool.

Library Build System
********************

1. The library build system shall be designed and implemented such that it can
be used directly and easily with spack for allowing users to build particular
flavors of the library.  This shall include build flags for specifying

   * the problem's dimension,
   * the Grid backend (required),
   * the runtime backend (including no backend),
   * if the Fortran interface shall be included in the library,
   * log/debug level.

2. The Runtime library can be installed into a directory specified at setup time
with the command line argument `--prefix`.

3. TODO: Should libraries be built with -fPIC/-fpic?

Test Build System
*****************

1. The Build Tool shall be a system that exists within the realm of the testing
portion of the repository and therefore outside the scope of the library.  It
shall provide users the necessary facilities to build and run tests.  A primary
function of the Build Tool shall be to build tests of the Runtime that are
provided in the repository as part of its continuous integration test suite. It
shall assume that the appropriate library to be tested by the specified test has
already been built and shall link against this.

2. The build tool shall determine the following flags based on command line input:

    * Test name
    * Dimensionality
    * Site directory
    * Build directory
    * Runtime parameter file ("par file")
    * Backend libraries
    * Debug levels
    * Multithreaded distributor

3. Users shall be required to specify a par file and the contents of the file
shall follow the JSON format.  The name of the file must be of the form
``<name>.json`` or ``<name>.json_base``.  An extension of ``json`` communicates
that the file is ready for immediate use; ``json_base``, that some of the
contents must be updated/altered before use.  The setup tool shall copy the
specified file to the build folder and shall always replace ``<name>`` with the
same base name, but without changing the extension.  This shall aid application
writers to always load runtime parameters from a file with a fixed name and to
setup job scripts that rerun the same test but with different parameters (e.g.,
as part of a performance study).

3. The build tool shall write a log file to the build directory that contains
the setup information and all metadata to reproduce the source tree at the time
of build (at minimum, commit number and a git diff). The log file can also
include extra information, like date and time, username, list of relevant
directories, etc.

4. The build tool shall automate the process of running a linter. The linter
shall always run before compilation.

5. The build tool shall allow users to request code coverage functionality be
built into the executable, although by default it is not. If code coverage is
requested, after the test is run an lcov code coverage report can be generated.

6. The build tool shall consist of information and components divided into three
categories, created and maintained by different actors:

    * Project-wide:
        * This category consists of information and components that are common
          across platforms, compilers, and tests. It also includes information
          that either (a) site managers and users cannot be expected to know or
          apply correctly and consistently, or (b) requires more intimate
          knowledge of Runtime implementation.
        * Some examples are:
            * List of files required to build different variants of the Runtime
            * library.
            * Compiler flags that are deemed necessary due to specific
              implementation details, as well as to ensure correct builds and
              execution.
            * Compiler flags for confirming tests adhere to a certain language
              standard.
        * It is intended that project-wide components and information be created
          and maintained by project maintainers, in accord with prevailing
          requirements and specifications.
    * Test-specific:
        * Test creators shall maintain test-specific contributions to the Build
          Tool and the test suite in accord with the prevailing requirements and
          specifications.  * Site-specific:
        * Site managers shall create and maintain site-specific build
          information; e.g. compiler flags, environment variables.

7. The Build Tool shall only require backend-specific information if the user is
trying to build a test or a library configuration that requires that specific
backend. Thus the tool shall not require site managers to support a backend they
cannot or do not intend to use.

8. **Do tests need to be linked with -fpie/-fPIE flags?**

9. Each test directory should include files named Flash_ND.h and constants_ND.h
where N is each supported dimensionality of the test. Each site should include
parameter files ("par files") which will be specified in a command line argument
and copied into the build directory as Flash_par.h.

10. The build system auto-generates files with the name Makefile.setup, Flash.h,
constants.h, and Flash_par.h in the build directory. Thus, files of these names
should not be located anywhere else in the repository.

11. In order to avoid errors caused by ambiguous include statements, the
following set of directories should not contain files with the same name:
	* The main source directores (currently `src` and `includes`)
	* The test directory of any given test
	* The site directory for any given site
Note that this means different tests and sites can (and should) have files with
overlapping names, for example Flash_ND.h files. This requirement is not checked
for in the current build system, so for now all contributors must self-enforce.

12. For every platform, site managers shall create and maintain a
`Makefile.site`, based on a template, which contains platform-specific flags
(e.g. optimization) and paths (e.g. AMReX). It need not include information
about backends that the test or library configuration does not require. Site
managers can refer to a template located in the sites base directory for
information on which flags are required.

13. Project maintainers shall maintain `Makefile.base` for the files in Runtime
and Grid units. It shall account for backed-specific variants.

14. Test creators shall maintain `Makefile.test` for each test with the files
specific to the test. 

15. There exists `Makefile` which gathers site-specific flags, contains other
flags based on command line input, and lists the make commands.

16. The build tool must create the build directory and empty it if it already
exists. The build directory's name will be specified from a command line
argument, and the folder will always be created in the repository's root
directory (which is determined by the location of the file setup.py).

17. The build tool copies parameter files (Flash.h, constants.h, and par file)
from the site and test directories.

18. The build tool symlinks all Makefiles into the build directory.

19. The build tool puts object files into a subdirectory of the build directory,
with a tree structure parallel to the source tree.

20. The build tool shall determine how to compile with the C++ 11 standard
depending on the compiler specified by the user (gnu, pgi, xl, or clang).

Grid Configuration
******************

1. Calling code shall be allowed to call ``load`` at most one time.  If any
configuration value stored in the configuration singleton when ``load`` is
called is invalid, then ``load`` shall throw an exception.

2. Calling code shall be allowed to call ``clear`` at most one time.  After
``clear`` is called, an exception shall be thrown if calling code attempts to
access the configuration singleton again or to call ``load``.  It shall be
acceptable for calling code to call ``clear`` without calling ``load``.

3. All Grid backend implementations shall call the configuration singleton's
``clear`` function immediately after consuming all configuration values.  It is
advised that the singleton, its values, and its ``clear`` function all be
accessed within a local block so that code in the same function cannot
accidentally access the singleton after ``clear`` is called.

The requirements imply that calling code could call ``load`` and terminate
without calling ``clear`` if the Grid singleton is never instantiated.  This is
acceptable since there is little risk of using the configuration values
inappropriately to the detriment of Grid execution.

Grid
****

    1. The Grid unit shall only allow calling code to call ``initDomain`` and
    ``destroyDomain`` at most once. The unit shall indicate an error if
    ``destroyDomain`` has been called but ``initDomain`` has not.  This is
    inline with the general program flow of our target domain's applications
    (but not our tests) and therefore presents a simpler, cleaner interface.

    2. Calling code shall pass to ``initDomain`` all information needed to setup
    the initial AMR grid structure and load the problem's initial conditions
    into the Grid unit's data structures.  This should allow for computing the
    initial conditions

       * in serial mode on the host using the tile iterator (with proper tiling allowed) and
       * using any thread team configuration available.

    This implies that the ``initBlock`` routine shall not be stored in the
    GridConfiguration singleton since an application might need to supply any
    number of variants of the ``initBlock`` routine for execution on different
    hardware.  The actual design of the ``initDomain`` interface as needed to
    accommodate the diversity of configuration values needed across different
    thread team configurations is not yet known.  I am presently happy to
    continue kicking that can down the road as it is related to the offline
    toolchain.  **Hopefully this requirement will allow for calling ``initDomain``
    in Fortran and supplying Fortran ``initBlock`` variants for use with the
    runtime**.

    3. The GridConfiguration class shall be an abstract, polymorphic, singleton
    class with one derived class per Grid backend (i.e., copy Tom's design for
    the Grid class).  The GridConfiguration public interface shall include an
    abstract ``load`` member function and a ``clear`` member function.  Since
    GridConfiguration is a singleton, it shall not be passed to Grid's
    ``instantiate``.  Rather, each Grid backend can choose what ``load`` needs to do
    before Grid's ``instantiate`` is called and what it should access/do directly
    in its constructor.  This requirement is motivated by the fact that
    ``amrex::AmrCore`` requires all AMReX configuration values to be loaded before
    instantiation.  Since the AMReX Grid backend will inherit from AmrCore, this
    loading into AMReX must be done before calling Grid's ``instantiate``.

    4. Each concrete Grid implementation shall call ``clear`` after consuming
    GridConfiguration data so that no other code can subsequently access the
    data.  **This is not enforceable**.

    5. Calling code shall be required to set in GridConfiguration function
    pointers to the application's ``errorEst`` routine.

    6. The Grid unit shall not include any notion of runtime parameters nor any
    facilities to load these.  Rather, it is the responsibility of calling code
    to configure the Grid unit with values by setting the values into the
    GridConfiguration instance.  This restriction reflects the reality that the
    Milhoja Grid interface does not care or need to know if its configuration
    values were fixed at compile time or loaded at runtime.  Also, it keeps the
    Milhoja interface small, simple, and clean.

AMReX Backend
-------------

    1. The AMReX Grid backend class shall be inherited from the abstract Grid and
    abstract amrex::AmrCore classes.  This does not simplify the public interface,
    but rather the implementation.  Now ownership of Grid configuration values need
    only be split between AMReX and this single Grid backend class.  In addition,
    loading/clearing of configuration values is simpler as both all Grid
    initialization occurs in the Grid AMReX backend class' constructor.  Therefore,
    there is no need for caching values nor managing cached values.  There is little
    risk in multiple inheritance as the Grid base class is _effectively_ an
    interface-only class as all functionality (outside of singleton instantiation
    and access) that it **presently** implements does not require or manage mutable
    state.  Therefore, the multiple inheritance is effectively combining interface
    from one class with interface/implementation of another, which is generally
    allowed in OOP languages developed after C++.  This weakening of the term
    "interface" based on mutable state, is rooted in Bjarne Stroustrup's C++11 book
    "The C++ Programming Language" (fourth edition) in section 21.3.1

        In fact, any class without mutable state can be used as an interface in
        a multiple-inheritance lattice without significant complications and
        overhead.  The key observation is that a class without mutable state can
        be replicated if necessary or shared if that is desired.

    2. The documentation in the Grid AMReX backend class' files shall state the
    need to maintain all implementations of functionality with no dependence on
    mutable state.

The AMReX backend has been designed as one main class, ``GridAMReX``, which is
derived from both the ``Grid`` class as well as ``amrex::AmrCore``.  While the
``Grid`` inheritance is sensibly public, we do not expose the inherited AMReX
interface.  This design was adopted over an earlier version of the class that
inherited only from ``Grid`` and used AmrCore as a mix-in.  Some tradeoffs were
made in this decision

* the implementation with multiple inheritance is perceived to be easier and
  cleaner,
* inheriting from AmrCore means that AMReX must configured and initialized prior
  to instantiating ``GridAMReX``, which drove the design of
  ``GridConfiguration`` including the need for the ``load`` member function,
* inheriting from AmrCore also meant that we needed to build in ``finalize`` to
  the polymorphic Grid singleton so that we can explicitly finalize AMReX before
  MPI is finalized,
* with ``finalize`` we finalize AMReX before finalizing ``amrex::AmrCore``,
  which is strange and ugly, and
* removing the indirection of the mix-in allows for direct access to principal
  data structures and avoids function calls, which might lead to a slightly
  improved performance.


Orchestration Runtime
*********************

    1. At instantiation, the runtime shall instantiate a given number, N, of
    distinct thread teams and each thread team shall be allowed to simultaneously
    use at most a given number, M, of threads.  Note that it is the client
    code's responsibility to determine M and N in accord with the runtime
    requirements and technical specifications presented here.

    2. Each thread team shall

       a. be created and run in the host CPU,
       b. be associated with a single MPI rank,
       c. be associated with a single unit of work (*e.g.*, tiles, blocks, or a data packet of blocks), and
       d. expose the same interface to client code regardless of the unit of work.

    3. For each execution cycle, a thread team shall be used by client code to
    apply at most one task (work or auxiliary) to a subset of the tiles managed
    by the team's associated MPI rank.  For the case of an auxiliary task, the
    subset is the empty set.  The restriction to one task will help make it
    easier to determine independence of tasks and teams.  For each cycle, the
    client code shall inform the thread team what task shall be executed and how
    many threads in the team should be activated immediately to start work on
    the task.  This implies that the task assigned to a particular thread team
    can change from one execution cycle to the next.

    4. Thread teams shall not need to know nor be informed of which device will
    carry out the computation associated with a given computational task.
    Rather the given computational task shall know where its block data resides
    in different memory and the task shall be written so that it can carry out
    its computations on the devices assigned to it.  This can include running
    code on the host CPU with the given team thread or using the team thread to
    launch computations on accelerator devices.  \Jared{This requirement is also
    related to data packets and will need improvement as the prototype evolves.}

    5. Each task to be run with a thread team shall have the same identical code
    interface so that task-specific information does not need to be passed to
    the task by the thread team.  This requirement helps decouple the thread
    team and therefore the runtime from the work being done by the thread teams.

This implies that client code must devise a scheme that makes all
task-/computation-specific parameters available to the function that defines the
task.  For FLASH, our present design is to implement all task functions as
routines in a unit and all such parameters as data members in the unit.  This
means that the code that calls the runtime will need to set the values of these
data members before the call.  For C++ tests of the runtime, task parameters
and task functions have been packaged up into a dedicated namespace so that they
are global but in an acceptable way.

    6. The thread team interface shall allow for client code to assign units of
    work to a thread team one unit at a time where the full work load given to a
    team during a single task execution is a subset of the blocks managed by the
    team's associated MPI rank.

    7. The thread team interface shall allow for client code to inform the team
    when all units of work to which the current task are to be applied have been
    given to the team.  This shall include the possibility of giving a thread
    team a task but no units of work.

    8. Client code shall trigger *via* the runtime interface a single runtime
    execution cycle that consists of executing potentially many distinct tasks
    (both auxiliary and work) on multiple different target devices.  The runtime
    interface shall provide the client code with a means to express what tasks
    are to be run as well as inter-task dependencies such that the runtime will
    be able to assemble an appropriate thread team configuration that does not
    violate the inter-task dependencies.  The runtime shall throw an error if
    the number of tasks in the bundle is more than the number of thread teams
    created by the runtime.

What does this look like?  The offline toolchain should determine the inter-task
dependencies, the mapping of tasks to HW, and the mapping of tasks to thread
teams.  Is the latter just choosing a thread team with the correct unit of work?
How does the toolchain specify to the runtime which thread team configuration to
use and the mapping of task to teams in the thread team configuration?  Can it
just be a long parameter list with multiple consecutive parameters in the list
specifying the task for a particular device?  The runtime could then see which
parameters specify a task and infer the thread team configuration from this.  We
would need, for instance, a CPU concurrent task, a GPU concurrent task, and a
post-GPU task.  How do you specify thread and work publishers/subscribers?

    9. The runtime shall contain a concurrent work distributor that facilitates
    applying multiple distinct tasks to all the blocks managed by the runtime's
    MPI rank.  Specifically, this distributor shall gather tiles using the Grid
    unit's tile iterator (or asynchronous tile iterator), form these into the
    appropriate units of work, and give the units of work to the appropriate
    thread teams.  Refer to Figure~\ref{fig:ConcurrentItor} for an example of
    such a scheme.

    10. The natural unit of data for a CPU is an appropriately sized tile.
    However, we suspect that a tile will be too little data to merit the
    overhead associated with launching a kernel.  Therefore, the Grid unit shall
    be retooled such that work distributors are capable of feeding tiles (proper
    subsets of blocks) to some thread teams and data packets of blocks to
    others

For AMReX, we might want to iterate over blocks and request a new iterator that
iterates over the tiles that cover a given block.

    11. The runtime shall contain a work splitting distributor that facilitates
    using more than one thread team to apply a single task to all the blocks
    managed by the runtime's MPI rank where the task is applied to each block by
    one and only one team.  Specifically, this distributor shall gather tiles
    using the Grid unit's tile iterator (or asynchronous tile iterator), use a
    distribution scheme to determine which tiles will be sent to which team,
    form these into the appropriate units of work based on the destination team,
    and send the units of work to the appropriate thread teams.  Refer to
    Figure~\ref{fig:SplitItor} for an example of such a scheme.

Allow for scheme that selects routing of work to team dynamically based on
current runtime telemetry data?

CUDA Backend
------------
