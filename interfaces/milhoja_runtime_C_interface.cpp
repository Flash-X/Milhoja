/** C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * This C interface does not consititute a true C interface.  Indeed, it was
 * designed to be used by the Fortran/C interoperability layer.  For example,
 * milhoja_runtime_init_c takes an MPI communicator in the form of an int, which
 * is intimately related to Fortran.  In addition, the C interface uses int
 * instead of unsigned int.  This constraint would not be necessary for a C
 * interface, but is required for use by the Fortran/C layer.
 *
 * Given this tight coupling to Fortran, the index sets made public in the C
 * interfaces (e.g. level, variable, and spatial) shall match those exposed in
 * the Fortran interface.  This could be advantageous as some arguments given to
 * this C interface could be pointers to Fortran variables.  Therefore, writing 
 * to Fortran variables directly here with correct index translations could lead
 * to better performance and simpler code in the Fortran/C interoperability
 * layer.
 *
 * Index set definitions
 *   - refinement level index set
 *        - [1, ..., MAX(int)] with level i being coarser than level i+1
 *   - UNK variables index set
 *        - [1, ..., MAX(int)]
 *   - Mesh element global spatial index set
 *        - Along any dimension in mesh that is <= NDIM,
 *             [NGUARD-1, nBlocks_i * nCells_i + NGUARD];
 *          otherwise, 1.  The full index set is therefore Ii x Ij x, Ik.
 *        - These are 1-based in the sense that the first interior cell
 *          is associated with index (1,1,1) as opposed to (0,0,0)
 *
 * Routines that accept arrays whose elements are associated with the
 * dimensions of the physical domain shall only read or write those elements
 * associated with the dimensions at or less than NDIM.  For those arguments
 * being written to or updated, calling code is, therefore, responsible for
 * setting those elements above NDIM as required by the calling code.  The code
 * in the C interface, cannot know how to set such values for all calling code.
 * This also implies that this interface does not assume nor require that such
 * arrays have sensible data above NDIM.
 *
 * In the interest of explicit, defensive programming and easing maintenance,
 * all casting from given types (e.g. int) to Milhoja types (e.g. unsigned int),
 * should occur in this layer and shall be written out explicitly, regardless of
 * correctness if handled by implicit casting.  All casts should be error
 * checked where necessary for overflow or loss of precision.  This layer does
 * not presume to know how the data will be used and therefore no sanity
 * checking of values shall be done here --- the error checking is just to
 * ensure correct casting.  Note that floating point variables shall appear in
 * function interfaces as Reals and not cast to float nor double.
 *
 * Each routine in the C interface shall be a function that returns an integer.
 * These functions shall return a zero value upon successful execution.  They
 * shall return a unique (across all functions in the complete C interface)
 * non-zero error code otherwise.
 *
 * All calls to Milhoja C++ code shall be wrapped in a try/catch block so that
 * any and all exceptions are caught and handled appropriately.  This mechanism
 * shall catch all exceptions based on std::exception so that the underlying
 * error message can be captured and made available to calling code.  It shall
 * also catch all other exceptions and handle these appropriately as well,
 * albeit without being able to provide calling code with a useful error
 * message.
 *
 * \todo Add definitions for how to deal with data above NDIM.  Should
 *       spatial index be 1?  deltas zero?  zMin/zMax?
 * 
 * \todo Much of this is the design of this layer and should, therefore, be
 * moved to the developer's guide or a dedicated design document.
 *
 * \todo Should we have something like orchestration::Integer and
 * orchestration::SizeT so that we match the use of orchestration::Real?
 *
 * \todo In some places, we use pointers to C++ classes as well as function
 * pointers declared using C++ syntax.  Also, data types such as Real are
 * in C++ namespaces.  Does any of this break C linkage/calling convention?  In
 * particular, this emphasizes the notion that this C interface is not intended
 * to be called from actual C code.
 *
 * \todo Should these error messages be cached in an error handling unit
 * so that calling code can access error code strings if they so choose?  Have a
 * file that links error code index to associated error message?  If so,
 * then no function in the C interface should write to stdout/stderr, which
 * makes good sense.
 */

#include <string>
#include <iostream>

#include <mpi.h>

#include "milhoja_interface_error_codes.h"
#include "OrchestrationLogger.h"
#include "actionRoutine.h"
#include "Runtime.h"

extern "C" {
    /**
     * \todo Error check given communicator and MPI_Comm_f2c call.
     * \todo Allow calling code to specify the log's filename.
     */
    int    milhoja_runtime_init_c(const int globalCommF,
                                  const int nThreadTeams, const int nThreadsPerTeam,
                                  const int nStreams,
                                  const size_t nBytesInMemoryPools) {
        const std::string   logFilename{"orchestration.log"};
    
        MPI_Comm    globalComm = MPI_Comm_f2c(globalCommF);

        if (nThreadTeams < 0) {
            std::cerr << "[milhoja_runtime_init_c] nThreadTeams is negative" << std::endl;
            return MILHOJA_ERROR_N_THREAD_TEAMS_NEGATIVE;
        } else if (nThreadsPerTeam < 0) {
            std::cerr << "[milhoja_runtime_init_c] nThreadsPerTeam is negative" << std::endl;
            return MILHOJA_ERROR_N_THREADS_PER_TEAM_NEGATIVE;
        } else if (nStreams < 0) {
            std::cerr << "[milhoja_runtime_init_c] nStreams is negative" << std::endl;
            return MILHOJA_ERROR_N_STREAMS_NEGATIVE;
        }

        // No possibility of overflow with int->unsigned int
        unsigned int   nThreadTeams_ui    = static_cast<unsigned int>(nThreadTeams);
        unsigned int   nThreadsPerTeam_ui = static_cast<unsigned int>(nThreadsPerTeam);
        unsigned int   nStreams_ui        = static_cast<unsigned int>(nStreams);

        try {
            orchestration::Logger::instantiate(globalComm, logFilename);
            orchestration::Runtime::instantiate(nThreadTeams_ui, nThreadsPerTeam_ui,
                                                nStreams_ui,
                                                nBytesInMemoryPools);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_runtime_init_c] Unable to initiate runtime\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_RUNTIME;
        } catch (...) {
            std::cerr << "[milhoja_runtime_init_c] Unable to initiate runtime\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_RUNTIME;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     *
     */
    int    milhoja_runtime_finalize_c(void) {
        return MILHOJA_SUCCESS;
    }

    /**
     * \todo Allow calling code to specify action name for improved logging.
     */
     int   milhoja_runtime_execute_tasks_cpu_c(orchestration::ACTION_ROUTINE taskFunction,
                                               const int nThreads) {
        using namespace orchestration;

        if (nThreads < 0) {
            std::cerr << "[milhoja_runtime_execute_tasks_cpu_c] nThreads is negative" << std::endl;
            return MILHOJA_ERROR_N_THREADS_NEGATIVE;
        }
        unsigned int nThreads_ui = static_cast<unsigned int>(nThreads);

        RuntimeAction     action;
        action.name            = "Lazy Action Name";
        action.nInitialThreads = nThreads_ui;
        action.teamType        = ThreadTeamDataType::BLOCK;
        action.nTilesPerPacket = 0;
        action.routine         = taskFunction;

        try {
            Runtime::instance().executeCpuTasks("Lazy Bundle Name", action);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_runtime_execute_tasks_cpu_c] Unable to execute tasks\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
        } catch (...) {
            std::cerr << "[milhoja_runtime_execute_tasks_cpu_c] Unable to execute tasks\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
        }

        return MILHOJA_SUCCESS;
     }
}
