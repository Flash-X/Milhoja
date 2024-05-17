/** C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * The C interfaces do not consititute true C interfaces.  Indeed, it was
 * designed to be used by the Fortran/C interoperability layer.  For example,
 * milhoja_grid_init_c takes an MPI communicator in the form of an int, which
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
 *          otherwise, 1.  The full index set is therefore Ii x Ij x Ik.
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
 * These functions shall return MILHOJA_SUCCESS upon successful execution.  They
 * shall return a unique (across all functions in the complete C interface)
 * error code different from MILHOJA_SUCCESS otherwise.
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
 * \todo Should we have something like milhoja::Integer and
 * milhoja::SizeT so that we match the use of milhoja::Real?
 *
 * \todo In some places, we use pointers to C++ classes as well as function
 * pointers declared using C++ syntax.  Also, data types such as Real are
 * in C++ namespaces.  Does any of this break C linkage/calling convention?  In
 * particular, this emphasizes the notion that this C interface is not intended
 * to be called from actual C code.
 */

#include <iostream>

#include "Milhoja.h"
#include "Milhoja_interface_error_codes.h"
#include "Milhoja_actionRoutine.h"
#include "Milhoja_Runtime.h"
#include "Milhoja_RuntimeBackend.h"
#include "Milhoja_TileWrapper.h"

extern "C" {
    /**
     * Perform all Milhoja initializations needed for calling code to begin
     * using its runtime infrastructure.  The Milhoja grid infrastructure must
     * be initialized before calling this routine.  See the documentation for
     * milhoja_grid_init_c for more information.
     *
     * \todo Allow calling code to specify the log's filename.
     *
     * \param nThreadTeams           The number of thread teams to use
     * \param nThreadsPerTeam        The number of threads to be assigned to each
     *                               team
     * \param nStreams               The number of streams to use
     * \param nBytesInCpuMemoryPool  The number of bytes to allocate in CPU memory
     *                               pool
     * \param nBytesInGpuMemoryPools The number of bytes to allocate in memory
     *                               pools associated with GPU (e.g., pinned &
     *                               GPU)
     *
     * \return The milhoja error code
     */
    int    milhoja_runtime_init_c(const int nThreadTeams, const int nThreadsPerTeam,
                                  const int nStreams,
                                  const size_t nBytesInCpuMemoryPool,
                                  const size_t nBytesInGpuMemoryPools) {
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

        unsigned int   nThreadTeams_ui    = static_cast<unsigned int>(nThreadTeams);
        unsigned int   nThreadsPerTeam_ui = static_cast<unsigned int>(nThreadsPerTeam);
        unsigned int   nStreams_ui        = static_cast<unsigned int>(nStreams);

        try {
            milhoja::Runtime::initialize(nThreadTeams_ui, nThreadsPerTeam_ui,
                                         nStreams_ui,
                                         nBytesInCpuMemoryPool,
                                         nBytesInGpuMemoryPools);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_RUNTIME;
        } catch (...) {
            std::cerr << "[milhoja_runtime_init_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_RUNTIME;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Finalize the runtime infrastructure.  Calling code should call this
     * routine before finalizing the grid infrastructure.
     *
     * \return The milhoja error code
     */
    int    milhoja_runtime_finalize_c(void) {
        try {
            milhoja::Runtime::instance().finalize();
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FINALIZE_RUNTIME;
        } catch (...) {
            std::cerr << "[milhoja_runtime_finalize_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FINALIZE_RUNTIME;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * The runtime presently has a rudimentary memory manager.  In order to run
     * simulations for a significant number of steps without exhausting GPU
     * memory, simulations must reset the runtime/memory manager at each step.
     * This is a temporary workaround until a proper memory manager is
     * implemented.
     *
     * \todo Remove once we have a proper runtime memory manager.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_reset_c(void) {
        try {
            milhoja::RuntimeBackend::instance().reset();
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_RESET_RUNTIME;
        } catch (...) {
            std::cerr << "[milhoja_runtime_reset_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_RESET_RUNTIME;
        }

        return MILHOJA_SUCCESS;
    }

#ifndef RUNTIME_MUST_USE_TILEITER
    /**
     * Instruct the runtime to set up the pipeline for the CPU-only thread team
     * configuration.
     *
     * \param taskFunction   The computational work to apply identically to each
     *                       block
     * \param nThreads       The number of threads that the single thread team
     *                       should activate.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_setup_pipeline_cpu_c(milhoja::ACTION_ROUTINE taskFunction,
                                              const int nThreads) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_setup_pipeline_cpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       }
       unsigned int    nThreads_ui = static_cast<unsigned int>(nThreads);

       milhoja::RuntimeAction     action;
       action.name            = "Lazy Action Name";
       action.nInitialThreads = nThreads_ui;
       action.teamType        = milhoja::ThreadTeamDataType::BLOCK;
       action.nTilesPerPacket = 0;
       action.routine         = taskFunction;

       try {
           milhoja::Runtime::instance().setupPipelineForCpuTasks("Lazy Bundle Name",
                                                        action);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_setup_pipeline_cpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }

    /**
     * Instruct the runtime to tear down the pipeline for the CPU-only thread team
     * configuration.
     *
     * \param nThreads       The number of threads that the single thread team
     *                       should have activated.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_teardown_pipeline_cpu_c(const int nThreads) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_teardown_pipeline_cpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       }

       try {
	 milhoja::Runtime::instance().teardownPipelineForCpuTasks("Lazy Bundle Name");
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_teardown_pipeline_cpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }

    /**
     * Push one tile to the prepared pipeline so that the thread team will
     * eventually execute the task.
     *
     * The computational work that is to apply identically to each block has been
     * established by the previous invocation of milhoja_runtime_setup_pipeline_cpu_c.
     *
     * \param tileWrapper    C pointer to the prototype TileWrapper
     * \param nThreads       The number of threads that the single thread team
     *                       should activate. Just for checking, not passed on from here.
     * \param tileInfo       Tile information, carries identity and properties of the
     *                       tile and links to the actual raw Flash-X real data.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_push_pipeline_cpu_c(void* tileWrapper,
                                              const int nThreads,
					      FlashxTileRaw* tileInfo) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_push_pipeline_cpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       }

       milhoja::TileWrapper*   prototype = static_cast<milhoja::TileWrapper*>(tileWrapper);

       try {
           milhoja::Runtime::instance().pushTileToPipeline("Lazy Bundle Name",
							   *prototype, tileInfo->sP, tileInfo->sI, tileInfo->sR);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_push_pipeline_cpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
#endif

#ifdef RUNTIME_CAN_USE_TILEITER
    /**
     * Instruct the runtime to use the CPU-only thread team configuration with
     * the given number of threads to apply the given task function to all
     * blocks.
     *
     * \todo Allow calling code to specify action name for improved logging.
     * \todo Need to add arguments for specifying the set of blocks.
     *
     * \param taskFunction   The computational work to apply identically to each
     *                       block
     * \param tileWrapper    C pointer to the prototype TileWrapper
     * \param nThreads       The number of threads that the single thread team
     *                       should activate.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_execute_tasks_cpu_c(milhoja::ACTION_ROUTINE taskFunction,
                                              void* tileWrapper,
                                              const int nThreads) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       }
       unsigned int    nThreads_ui = static_cast<unsigned int>(nThreads);

       milhoja::RuntimeAction     action;
       action.name            = "Lazy Action Name";
       action.nInitialThreads = nThreads_ui;
       action.teamType        = milhoja::ThreadTeamDataType::BLOCK;
       action.nTilesPerPacket = 0;
       action.routine         = taskFunction;

       milhoja::TileWrapper*   prototype = static_cast<milhoja::TileWrapper*>(tileWrapper);

       try {
           milhoja::Runtime::instance().executeCpuTasks("Lazy Bundle Name",
                                                        action, *prototype);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
#endif

#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifndef RUNTIME_MUST_USE_TILEITER
    /**
     * Set up the runtime to use the GPU-only thread team configuration with
     * the given number of threads to apply the given task function to all
     * blocks.
     *
     * \todo Allow calling code to specify action name for improved logging.
     * \todo Add stagger as an argument if it proves useful.
     *
     * \param taskFunction          The computational work to apply identically
     *                              to each block
     * \param nThreads              The number of threads that the single thread
     *                              team should activate.
     * \param nTilesPerPacket       The maximum number of tiles allowed in each
     *                              packet
     * \param packet                Pointer to a prototype data packet to be
     *                              used to create new packets.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_setup_pipeline_gpu_c(milhoja::ACTION_ROUTINE taskFunction,
                                              const int nThreads,
                                              const int nTilesPerPacket,
                                              void* packet) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_setup_pipeline_gpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) {
           std::cerr << "[milhoja_runtime_setup_pipeline_gpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       unsigned int    stagger_usec_ui        = 0;
       unsigned int    nThreads_ui            = static_cast<unsigned int>(nThreads);
       unsigned int    nTilesPerPacket_ui     = static_cast<unsigned int>(nTilesPerPacket);

       milhoja::DataPacket*   prototype = static_cast<milhoja::DataPacket*>(packet);

       milhoja::RuntimeAction     action;
       action.name            = "Lazy GPU setup Action Name";
       action.nInitialThreads = nThreads_ui;
       action.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
       action.nTilesPerPacket = nTilesPerPacket_ui;
       action.routine         = taskFunction;

       try {
           milhoja::Runtime::instance().setupPipelineForGpuTasks("Lazy GPU Bundle Name",
                                                        stagger_usec_ui,
                                                        action,
                                                        *prototype);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_setup_pipeline_gpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }
    int   milhoja_runtime_setup_pipeline_cpugpu_c(milhoja::ACTION_ROUTINE pktTaskFunction,
                                              milhoja::ACTION_ROUTINE tileTaskFunction,
                                              const int nThreads,
                                              const int nTilesPerPacket,
                                              void* packet) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_setup_pipeline_cpugpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) {
           std::cerr << "[milhoja_runtime_setup_pipeline_cpugpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       unsigned int    nThreads_ui            = static_cast<unsigned int>(nThreads);
       unsigned int    nTilesPerPacket_ui     = static_cast<unsigned int>(nTilesPerPacket);

       milhoja::DataPacket*   prototype = static_cast<milhoja::DataPacket*>(packet);

       milhoja::RuntimeAction     pktAction;
       pktAction.name            = "Lazy GPU setup Action Name";
       pktAction.nInitialThreads = nThreads_ui;
       pktAction.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
       pktAction.nTilesPerPacket = nTilesPerPacket_ui;
       pktAction.routine         = pktTaskFunction;

       milhoja::RuntimeAction     tileAction;
       tileAction.name            = "Lazy CPU setup Action Name";
       tileAction.nInitialThreads = nThreads_ui;
       tileAction.teamType        = milhoja::ThreadTeamDataType::BLOCK;
       tileAction.nTilesPerPacket = 0;
       tileAction.routine         = tileTaskFunction;

       try {
           milhoja::Runtime::instance().setupPipelineForCpuGpuTasks("Lazy CPUGPU Bundle Name",
                                                        pktAction,
                                                        tileAction,
                                                        *prototype);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_setup_pipeline_cpugpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }
    int   milhoja_runtime_setup_pipeline_extgpu_c(milhoja::ACTION_ROUTINE taskFunction,
                                              milhoja::ACTION_ROUTINE postTaskFunction,
                                              const int nThreads,
                                              const int nTilesPerPacket,
                                              void* packet,
                                              void* tileWrapper) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_setup_pipeline_extgpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) {
           std::cerr << "[milhoja_runtime_setup_pipeline_extgpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       unsigned int    nThreads_ui            = static_cast<unsigned int>(nThreads);
       unsigned int    nTilesPerPacket_ui     = static_cast<unsigned int>(nTilesPerPacket);

       milhoja::TileWrapper*  tilePrototype   = static_cast<milhoja::TileWrapper*>(tileWrapper);
       milhoja::DataPacket*   prototype = static_cast<milhoja::DataPacket*>(packet);

       milhoja::RuntimeAction     pktAction;
       pktAction.name            = "Lazy GPU setup Action Name";
       pktAction.nInitialThreads = nThreads_ui;
       pktAction.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
       pktAction.nTilesPerPacket = nTilesPerPacket_ui;
       pktAction.routine         = taskFunction;

       milhoja::RuntimeAction     postAction;
       postAction.name            = "Lazy CPU setup Action Name";
       postAction.nInitialThreads = nThreads_ui;
       postAction.teamType        = milhoja::ThreadTeamDataType::BLOCK;
       postAction.nTilesPerPacket = 0;
       postAction.routine         = postTaskFunction;

       try {
           milhoja::Runtime::instance().setupPipelineForExtGpuTasks("Lazy EXTGPU Bundle Name",
                                                        pktAction,
                                                        postAction,
                                                        *prototype,
                                                        *tilePrototype);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_setup_pipeline_extgpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_SETUP_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }
    int   milhoja_runtime_teardown_pipeline_gpu_c(const int nThreads,
                                              const int nTilesPerPacket) {
       if (nThreads < 0) {      // nThreads: only use in this function
           std::cerr << "[milhoja_runtime_teardown_pipeline_gpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) { // nTilesPerPacket: only use here
           std::cerr << "[milhoja_runtime_teardown_pipeline_gpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       try {
           milhoja::Runtime::instance().teardownPipelineForGpuTasks("Lazy GPU setup Bundle Name");
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_teardown_pipeline_gpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }
    int   milhoja_runtime_teardown_pipeline_cpugpu_c(const int nThreads,
                                              const int nTilesPerPacket) {
       if (nThreads < 0) {      // nThreads: only use in this function
           std::cerr << "[milhoja_runtime_teardown_pipeline_cpugpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) { // nTilesPerPacket: only use here
           std::cerr << "[milhoja_runtime_teardown_pipeline_cpugpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       try {
           milhoja::Runtime::instance().teardownPipelineForCpuGpuTasks("Lazy CPUGPU setup Bundle Name");
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_teardown_pipeline_cpugpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }
    int   milhoja_runtime_teardown_pipeline_extgpu_c(const int nThreads,
                                              const int nTilesPerPacket) {
       if (nThreads < 0) {      // nThreads: only use in this function
           std::cerr << "[milhoja_runtime_teardown_pipeline_extgpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) { // nTilesPerPacket: only use here
           std::cerr << "[milhoja_runtime_teardown_pipeline_extgpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       try {
           milhoja::Runtime::instance().teardownPipelineForExtGpuTasks("Lazy EXTGPU setup Bundle Name");
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       } catch (...) {
           std::cerr << "[milhoja_runtime_teardown_pipeline_extgpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_TEARDOWN_PIPELINE;
       }

       return MILHOJA_SUCCESS;
    }
    /**
     * Push one tile to the prepared pipeline so that the thread team will
     * eventually execute the task.
     *
     * The computational work to be applied to each tile is expected to have been
     * established by the previous invocation of milhoja_runtime_setup_pipeline_gpu_c.
     *
     * \param packet         Pointer to a prototype data packet to be
     *                       used to create new packets.
     * \param nThreads       The number of threads that the single thread team
     *                       should activate. This is here just for checking, and
     *                       is not passed on down from here.
     * \param tileInfo       Tile information, carries identity and properties of the
     *                       tile and links to the actual raw Flash-X real data.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_push_pipeline_gpu_c(void* packet,
					      const int nThreads,
                                              FlashxTileRaw* tileInfo) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_push_pipeline_gpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       }

       milhoja::DataPacket*   prototype = static_cast<milhoja::DataPacket*>(packet);

       try {
           milhoja::Runtime::instance().pushTileToGpuPipeline("Lazy Bundle Name",
							   *prototype, tileInfo->sP, tileInfo->sI, tileInfo->sR);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_push_pipeline_gpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
    int   milhoja_runtime_push_pipeline_cpugpu_c(void* tileWrapper,
                                                 void* packet,
                                                 const int nThreads,
                                                 FlashxTileRaw* tileInfo) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_push_pipeline_cpugpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       }

       milhoja::TileWrapper*  tilePrototype  = static_cast<milhoja::TileWrapper*>(tileWrapper);
       milhoja::DataPacket*   prototype      = static_cast<milhoja::DataPacket*>(packet);

       try {
           milhoja::Runtime::instance().pushTileToCpuGpuPipeline("Lazy Bundle Name",
                                                           *tilePrototype,
                                                           *prototype, tileInfo->sP, tileInfo->sI, tileInfo->sR);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_push_pipeline_cpugpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
    int   milhoja_runtime_push_pipeline_extgpu_c(void* packet,
					      const int nThreads,
                                              FlashxTileRaw* tileInfo) {
       if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_push_pipeline_extgpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       }

       milhoja::DataPacket*   prototype = static_cast<milhoja::DataPacket*>(packet);

       try {
           milhoja::Runtime::instance().pushTileToExtGpuPipeline("Lazy Bundle Name",
							   *prototype, tileInfo->sP, tileInfo->sI, tileInfo->sR);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_push_pipeline_extgpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
#  endif

#  ifdef RUNTIME_CAN_USE_TILEITER
    /**
     * Instruct the runtime to use the GPU-only thread team configuration with
     * the given number of threads to apply the given task function to all
     * blocks.
     *
     * \todo Allow calling code to specify action name for improved logging.
     * \todo Need to add arguments for specifying the set of blocks.
     * \todo Add stagger as an argument if it proves useful.
     *
     * \param taskFunction          The computational work to apply identically
     *                              to each block
     * \param nDistributorThreads   The number of distributor threads to use
     * \param nThreads              The number of threads that the single thread
     *                              team should activate.
     * \param nTilesPerPacket       The maximum number of tiles allowed in each
     *                              packet
     * \param packet                Pointer to a prototype data packet to be
     *                              used to create new packets.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_execute_tasks_gpu_c(milhoja::ACTION_ROUTINE taskFunction,
                                              const int nDistributorThreads,
                                              const int nThreads,
                                              const int nTilesPerPacket,
                                              void* packet) {
       if        (nDistributorThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_gpu_c] nDistributorThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_gpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_gpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       unsigned int    nDistributorThreads_ui = static_cast<unsigned int>(nDistributorThreads);
       unsigned int    stagger_usec_ui        = 0;
       unsigned int    nThreads_ui            = static_cast<unsigned int>(nThreads);
       unsigned int    nTilesPerPacket_ui     = static_cast<unsigned int>(nTilesPerPacket);

       milhoja::DataPacket*   prototype = static_cast<milhoja::DataPacket*>(packet);

       milhoja::RuntimeAction     action;
       action.name            = "Lazy GPU Action Name";
       action.nInitialThreads = nThreads_ui;
       action.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
       action.nTilesPerPacket = nTilesPerPacket_ui;
       action.routine         = taskFunction;

       try {
           milhoja::Runtime::instance().executeGpuTasks("Lazy GPU Bundle Name",
                                                        nDistributorThreads_ui,
                                                        stagger_usec_ui,
                                                        action,
                                                        *prototype);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_execute_tasks_gpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
    /**
     * Instruct the runtime to use the CPU/GPU thread team configuration with
     * the given number of threads to apply the given "CPU" task function to all
     * blocks in direct (tile-wrapped) form ("on the CPU"), and the "GPU"
     * task function in packet form ("on the GPU").
     *
     * \todo Allow calling code to specify action name for improved logging.
     * \todo Should add arguments for specifying the set of blocks.
     * \todo Add stagger as an argument if it proves useful.
     *
     * \param cpuTaskFunction       The computational work to apply identically
     *                              to each block "on the CPU"
     * \param gpuTaskFunction       The computational work to apply identically
     *                              to each block "on the GPU"
     * \param nDistributorThreads   The number of distributor threads to use
     *                              (may be ignored)
     * \param nThreads              The number of threads that the single thread
     *                              team should activate.
     * \param nTilesPerPacket       The maximum number of tiles allowed in each
     *                              packet
     * \param packet                Pointer to a prototype data packet to be
     *                              used to create new packets.
     * \param tileWrapper           Pointer to a prototype tile wrapper to be
     *                              used to create new tileWrapper objects to enqueue.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_execute_tasks_cpugpu_c(milhoja::ACTION_ROUTINE cpuTaskFunction,
                                                 milhoja::ACTION_ROUTINE gpuTaskFunction,
                                                 const int nDistributorThreads,
                                                 const int nThreads,
                                                 const int nTilesPerPacket,
                                                 void* packet,
                                                 void* tileWrapper) {
       if        (nDistributorThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpu_c] nDistributorThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       unsigned int    nThreads_ui            = static_cast<unsigned int>(nThreads);
       unsigned int    nTilesPerPacket_ui     = static_cast<unsigned int>(nTilesPerPacket);

       milhoja::TileWrapper*  tilePrototype   = static_cast<milhoja::TileWrapper*>(tileWrapper);
       milhoja::DataPacket*   prototype       = static_cast<milhoja::DataPacket*>(packet);

       milhoja::RuntimeAction     pktAction;
       pktAction.name            = "Lazy GPU Action Name";
       pktAction.nInitialThreads = nThreads_ui;
       pktAction.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
       pktAction.nTilesPerPacket = nTilesPerPacket_ui;
       pktAction.routine         = gpuTaskFunction;

       milhoja::RuntimeAction     cpuAction;
       cpuAction.name            = "Lazy CPU Action Name";
       cpuAction.nInitialThreads = nThreads_ui;
       cpuAction.teamType        = milhoja::ThreadTeamDataType::BLOCK;
       cpuAction.nTilesPerPacket = 0;
       cpuAction.routine         = cpuTaskFunction;

       try {
           milhoja::Runtime::instance().executeCpuGpuTasks("Lazy CPUGPU Bundle Name",
                                                        cpuAction,
                                                        *tilePrototype,
                                                        pktAction,
                                                        *prototype);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
    /**
     * Instruct the runtime to use the Split CPU/GPU thread team configuration
     * with the given number of threads to apply the given packet task
     * function to some of the blocks in packet form ("on the GPU") and
     * and the tile task function in direct (tile-wrapped) form ("on the CPU")
     * to the other blocks, in an alternating fashion.
     *
     * \todo Allow calling code to specify action name for improved logging.
     * \todo Should add arguments for specifying the set of blocks.
     * \todo Add stagger as an argument if it proves useful.
     *
     * \param cpuTaskFunction       The computational work to apply identically
     *                              to some blocks "on the CPU"
     * \param gpuTaskFunction       The computational work to apply identically
     *                              to other blocks "on the GPU"
     * \param nDistributorThreads   The number of distributor threads to use
     *                              (may be ignored)
     * \param nThreads              The number of threads that the single thread
     *                              team should activate.
     * \param nTilesPerPacket       The maximum number of tiles allowed in each
     *                              packet
     * \param packet                Pointer to a prototype data packet to be
     *                              used to create new packets.
     * \param tileWrapper           Pointer to a prototype tile wrapper to be
     *                              used to create new tileWrapper objects to enqueue.
     * \param nTilesPerCpuTurn      Each CpuGpuSplit distributor thread will hand the first
     *                              `nTilesPerCpuTurn` tiles from the grid iterator to
     *                              the cpuTaskFunction (as separate tasks), then build
     *                              a data packet from the next `nTilesPerPacket` tiles
     *                              and - when full - hand it to the gpuTaskFunction as
     *                              one task; then repeat; until the iterator is exhausted.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_execute_tasks_cpugpusplit_c(milhoja::ACTION_ROUTINE cpuTaskFunction,
                                                 milhoja::ACTION_ROUTINE gpuTaskFunction,
                                                 const int nDistributorThreads,
                                                 const int nThreads,
                                                 const int nTilesPerPacket,
                                                 const int nTilesPerCpuTurn,
                                                 void* packet,
                                                 void* tileWrapper) {
       if        (nDistributorThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpusplit_c] nDistributorThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpusplit_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpusplit_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       unsigned int    nDistributorThreads_ui = static_cast<unsigned int>(nDistributorThreads);
       unsigned int    stagger_usec_ui        = 0;
       unsigned int    nThreads_ui            = static_cast<unsigned int>(nThreads);
       unsigned int    nTilesPerPacket_ui     = static_cast<unsigned int>(nTilesPerPacket);
       unsigned int    nTilesPerCpuTurn_ui    = static_cast<unsigned int>(nTilesPerCpuTurn);

       milhoja::TileWrapper*  tilePrototype   = static_cast<milhoja::TileWrapper*>(tileWrapper);
       milhoja::DataPacket*   prototype       = static_cast<milhoja::DataPacket*>(packet);

       milhoja::RuntimeAction     pktAction;
       pktAction.name            = "Lazy GPU Action Name";
       pktAction.nInitialThreads = nThreads_ui;
       pktAction.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
       pktAction.nTilesPerPacket = nTilesPerPacket_ui;
       pktAction.routine         = gpuTaskFunction;

       milhoja::RuntimeAction     cpuAction;
       cpuAction.name            = "Lazy CPU Action Name";
       cpuAction.nInitialThreads = nThreads_ui;
       cpuAction.teamType        = milhoja::ThreadTeamDataType::BLOCK;
       cpuAction.nTilesPerPacket = 0;
       cpuAction.routine         = cpuTaskFunction;

       try {
           milhoja::Runtime::instance().executeCpuGpuSplitTasks("Split CPUGPU Bundle Name",
                                                                nDistributorThreads_ui,
                                                                stagger_usec_ui,
                                                                cpuAction,
                                                                *tilePrototype,
                                                                pktAction,
                                                                *prototype,
                                                                nTilesPerCpuTurn_ui);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_execute_tasks_cpugpusplit_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
    /**
     * Instruct the runtime to use the GPU/post-GPU thread team configuration with
     * the given number of threads to apply the given task function to all
     * blocks in packet form ("on the GPU"), followed by the "Post" task function
     * in direct (tile-wrapped) form ("on the CPU").
     *
     * \todo Allow calling code to specify action name for improved logging.
     * \todo Should add arguments for specifying the set of blocks.
     * \todo Add stagger as an argument if it proves useful.
     *
     * \param taskFunction          The computational work to apply identically
     *                              to each block "on the GPU"
     * \param postTaskFunction      The computational work to apply identically
     *                              to each block "on the CPU"
     * \param nDistributorThreads   The number of distributor threads to use
     * \param nThreads              The number of threads that the single thread
     *                              team should activate.
     * \param nTilesPerPacket       The maximum number of tiles allowed in each
     *                              packet
     * \param packet                Pointer to a prototype data packet to be
     *                              used to create new packets.
     *
     * \return The milhoja error code
     */
    int   milhoja_runtime_execute_tasks_extgpu_c(milhoja::ACTION_ROUTINE taskFunction,
                                              milhoja::ACTION_ROUTINE postTaskFunction,
                                              const int nDistributorThreads,
                                              const int nThreads,
                                              const int nTilesPerPacket,
                                              void* packet,
                                              void* tileWrapper) {
       if        (nDistributorThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_extgpu_c] nDistributorThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nThreads < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_extgpu_c] nThreads is negative" << std::endl;
           return MILHOJA_ERROR_N_THREADS_NEGATIVE;
       } else if (nTilesPerPacket < 0) {
           std::cerr << "[milhoja_runtime_execute_tasks_extgpu_c] nTilesPerPacket is negative" << std::endl;
           return MILHOJA_ERROR_N_TILES_NEGATIVE;
       }

       unsigned int    nDistributorThreads_ui = static_cast<unsigned int>(nDistributorThreads);
       unsigned int    nThreads_ui            = static_cast<unsigned int>(nThreads);
       unsigned int    nTilesPerPacket_ui     = static_cast<unsigned int>(nTilesPerPacket);

       milhoja::TileWrapper*  tilePrototype   = static_cast<milhoja::TileWrapper*>(tileWrapper);
       milhoja::DataPacket*   prototype = static_cast<milhoja::DataPacket*>(packet);

       milhoja::RuntimeAction     pktAction;
       pktAction.name            = "Lazy GPU Action Name";
       pktAction.nInitialThreads = nThreads_ui;
       pktAction.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
       pktAction.nTilesPerPacket = nTilesPerPacket_ui;
       pktAction.routine         = taskFunction;

       milhoja::RuntimeAction     postAction;
       postAction.name            = "Lazy CPU Action Name";
       postAction.nInitialThreads = nThreads_ui;
       postAction.teamType        = milhoja::ThreadTeamDataType::BLOCK;
       postAction.nTilesPerPacket = 0;
       postAction.routine         = postTaskFunction;

       try {
           milhoja::Runtime::instance().executeExtendedGpuTasks("Lazy GPU Bundle Name",
                                                        nDistributorThreads_ui,
                                                        pktAction,
                                                        postAction,
                                                        *prototype,
                                                        *tilePrototype);
       } catch (const std::exception& exc) {
           std::cerr << exc.what() << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       } catch (...) {
           std::cerr << "[milhoja_runtime_execute_tasks_extgpu_c] Unknown error caught" << std::endl;
           return MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS;
       }

       return MILHOJA_SUCCESS;
    }
#  endif
#endif   // #ifdef RUNTIME_SUPPORT_DATAPACKETS
}

