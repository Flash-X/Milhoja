Requirements
============
The statements made in this section are intended to be informal, high-level
requirements.  Specifically, they should be correct for all orchestration
runtime designs and ideally require little change if our design changes
significantly.  In other words, these statements should not be informed by
implementation ideas or details.

System
******

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

    3. The AMReX Grid backend class shall be inherited from the abstract Grid and
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

    4. The documentation in the Grid AMReX backend class' files shall state the
    need to maintain all implementations of functionality with no dependence on
    mutable state.

    5. The GridConfiguration class shall be an abstract, polymorphic, singleton
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

    6. Each concrete Grid implementation shall call ``clear`` after consuming
    GridConfiguration data so that no other code can subsequently access the
    data.  **This is not enforceable**.

    7. Calling code shall be required to set in GridConfiguration function
    pointers to the application's ``errorEst`` routine.

    8. The Grid unit shall not include any notion of runtime parameters nor any
    facilities to load these.  Rather, it is the responsibility of calling code
    to configure the Grid unit with values by setting the values into the
    GridConfiguration instance.  This restriction reflects the reality that the
    Milhoja Grid interface does not care or need to know if its configuration
    values were fixed at compile time or loaded at runtime.  Also, it keeps the
    Milhoja interface small, simple, and clean.

AMReX Backend
-------------

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
