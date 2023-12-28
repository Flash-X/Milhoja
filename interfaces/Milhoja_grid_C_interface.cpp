/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in Milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <vector>
#include <string>
#include <iostream>

#include <mpi.h>

#include "Milhoja.h"
#include "Milhoja_coordinateSystem.h"
#include "Milhoja_boundaryConditions.h"
#include "Milhoja_Logger.h"
#include "Milhoja_GridConfiguration.h"
#include "Milhoja_Grid.h"
#include "Milhoja_interface_error_codes.h"
#include "Milhoja_actionRoutine.h"
#include "Milhoja_RuntimeAction.h"

extern "C" {
    /**
     * Perform all Milhoja initializations needed for calling code to begin
     * using its grid infrastructure.  Note that calling code must subsequently
     * call Grid_initDomain when deemed appropriate to setup the problem domain
     * in the grid infrastructure.  Calling code must initialize MPI before
     * calling this routine.
     *
     * Note that the Milhoja grid infrastructure can be used without using the
     * runtime infrastructure.  However, the runtime cannot be used without the
     * grid infrastructure.  Therefore, initializing the grid also initializes
     * the general Milhoja infrastructure.  As a result, the grid must be
     * initialized before the runtime; the runtime must be finalized before the
     * grid.  This also makes conceptual sense since the runtime depends on the
     * grid infrastructure.
     *
     * \todo Milhoja_GridConfigurationAMReX presently expects
     * maxRefinementLevel to be 1-based.  Is this really what the C++ code
     * wants?
     *
     * \param globalCommF          The Fortran version of the MPI communicator that
     *                             Milhoja should use
     * \param logRank              The rank in the given communicator of the MPI process
     *                             that should perform logging duties.
     * \param coordSys             The domain's coordinate system.  See
     *                             Milhoja.h for valid values.
     * \param xMin                 Define the physical domain in X as [xMin, xMax]
     * \param xMax                 See xMin
     * \param yMin                 Define the physical domain in Y as [yMin, yMax]
     * \param yMax                 See yMin
     * \param zMin                 Define the physical domain in Z as [zMin, zMax]
     * \param zMax                 See zMin
     * \param loBCs                Indicate how BCs should be handled at each of
     *                             the low domain faces.  See Milhoja.h for
     *                             valid values.
     * \param externalBcRoutine    Function that grid backend will use to set
     *                             non-periodic boundary condition data.
     * \param hiBCs                High face version of loBCs
     * \param nxb                  The number of cells along X in each block in the
     *                             domain decomposition
     * \param nyb                  The number of cells along Y in each block in the
     *                             domain decomposition
     * \param nzb                  The number of cells along Z in each block in the
     *                             domain decomposition
     * \param nBlocksX             The number of blocks along X in the domain decomposition
     * \param nBlocksY             The number of blocks along Y in the domain decomposition
     * \param nBlocksZ             The number of blocks along Z in the domain decomposition
     * \param maxRefinementLevel   The 1-based index of the finest refinement level
     *                             permitted at any time during the simulation
     * \param ccInterpolator       The interpolator to use for cell-centered
     *                             data.  See Milhoja.h for macros &
     *                             Milhoja_interpolator.h for more information.
     * \param nGuard               The number of guardcells
     * \param nCcVars              The number of physical variables in the solution
     * \param nFluxVars            The number of flux variables needed
     * \param errorEst             Procedure that is used to assess if a block should
     *                             be refined, derefined, or stay at the same
     *                             refinement
     * \return The milhoja error code
     */
    int    milhoja_grid_init_c(const MPI_Fint globalCommF,
                               const int logRank,
                               const int coordSys,
                               const milhoja::Real xMin, const milhoja::Real xMax,
                               const milhoja::Real yMin, const milhoja::Real yMax,
                               const milhoja::Real zMin, const milhoja::Real zMax,
                               const int* loBCs, const int* hiBCs,
                               milhoja::BC_ROUTINE externalBcRoutine,
                               const int nxb, const int nyb, const int nzb,
                               const int nBlocksX,
                               const int nBlocksY,
                               const int nBlocksZ,
                               const int maxRefinementLevel,
                               const int ccInterpolator,
                               const int nGuard,
                               const int nCcVars, const int nFluxVars,
                               milhoja::ERROR_ROUTINE errorEst) {
        MPI_Comm    globalComm = MPI_Comm_f2c(globalCommF);

        if (   (nxb      < 0) || (nyb      < 0) || (nzb      < 0) 
            || (nBlocksX < 0) || (nBlocksY < 0) || (nBlocksZ < 0)
            || (maxRefinementLevel < 0) || (nGuard < 0)
            || (nCcVars < 0) || (nFluxVars < 0)) {
            std::cerr << "[milhoja_grid_init_c] Invalid configuration value" << std::endl;
            return MILHOJA_ERROR_NEGATIVE_VALUE_FOR_UINT;
        } else if (!loBCs || !hiBCs) {
            std::cerr << "[milhoja_grid_init_c] BCs null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }
        unsigned int    nxb_ui                = static_cast<unsigned int>(nxb);
        unsigned int    nyb_ui                = static_cast<unsigned int>(nyb);
        unsigned int    nzb_ui                = static_cast<unsigned int>(nzb);
        unsigned int    nBlocksX_ui           = static_cast<unsigned int>(nBlocksX);
        unsigned int    nBlocksY_ui           = static_cast<unsigned int>(nBlocksY);
        unsigned int    nBlocksZ_ui           = static_cast<unsigned int>(nBlocksZ);
        unsigned int    maxRefinementLevel_ui = static_cast<unsigned int>(maxRefinementLevel);
        unsigned int    nGuard_ui             = static_cast<unsigned int>(nGuard);
        unsigned int    nCcVars_ui            = static_cast<unsigned int>(nCcVars);
        unsigned int    nFluxVars_ui          = static_cast<unsigned int>(nFluxVars);

        try {
            milhoja::Logger::initialize("milhoja.log", globalComm, logRank);

            // Configure in local block so that code cannot accidentally access
            // configuration data after it is consumed by Grid at initialization.
            milhoja::GridConfiguration&   cfg = milhoja::GridConfiguration::instance();

            cfg.coordSys          = static_cast<milhoja::CoordSys>(coordSys); 
            cfg.xMin              = xMin;
            cfg.xMax              = xMax;
            cfg.yMin              = yMin;
            cfg.yMax              = yMax;
            cfg.zMin              = zMin;
            cfg.zMax              = zMax;
            for (unsigned int i=0; i<MILHOJA_MDIM; ++i) {
                cfg.loBCs[i] = static_cast<milhoja::BCs>(loBCs[i]);
                cfg.hiBCs[i] = static_cast<milhoja::BCs>(hiBCs[i]);
            }
#ifdef FULL_MILHOJAGRID
	    cfg.externalBcRoutine = externalBcRoutine;
#endif
            cfg.nxb               = nxb_ui;
            cfg.nyb               = nyb_ui;
            cfg.nzb               = nzb_ui;
            cfg.nCcVars           = nCcVars_ui;
            cfg.nFluxVars         = nFluxVars_ui;
            cfg.nGuard            = nGuard_ui;
            cfg.nBlocksX          = nBlocksX_ui;
            cfg.nBlocksY          = nBlocksY_ui;
            cfg.nBlocksZ          = nBlocksZ_ui;
            cfg.maxFinestLevel    = maxRefinementLevel_ui;
#ifdef FULL_MILHOJAGRID
	    cfg.errorEstimation   = errorEst;
#endif
            cfg.ccInterpolator    = static_cast<milhoja::Interpolator>(ccInterpolator);
            cfg.mpiComm           = globalComm;
   
            cfg.load();
            milhoja::Grid::initialize();
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_GRID;
        } catch (...) {
            std::cerr << "[milhoja_grid_init_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_GRID;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Finalize the grid infrastructure.  It is assumed that calling code is
     * responsible for finalizing MPI and does so *after* calling this routine.
     *
     * Calling code should finalize the grid before finalizing the runtime.  See
     * the documentation for milhoja_grid_init_c for more details.
     *
     * \return The milhoja error code
     */
    int    milhoja_grid_finalize_c(void) {
        try {
#ifdef FULL_MILHOJAGRID
            milhoja::Grid::instance().destroyDomain();
#endif
            milhoja::Grid::instance().finalize();
            milhoja::Logger::instance().finalize();
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FINALIZE_GRID;
        } catch (...) {
            std::cerr << "[milhoja_grid_finalize_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FINALIZE_GRID;
        }

        return MILHOJA_SUCCESS;
    }

#ifdef FULL_MILHOJAGRID
    /**
     * Obtain the coordinate system used to define the domain.
     *
     * \param system   The pointer whose variable will store the system type
     * \return The milhoja error code
     */
    int   milhoja_grid_coordinate_system_c(int* system) {
        if (!system) {
            std::cerr << "[milhoja_grid_coordinate_system_c] system is null" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            milhoja::Grid&      grid = milhoja::Grid::instance();
            *system = static_cast<int>(grid.getCoordinateSystem());
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_COORD_SYS;
        } catch (...) {
            std::cerr << "[milhoja_grid_coordinate_system_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_COORD_SYS;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the low and high coordinates in physical space of the rectangular
     * box that bounds the problem's spatial domain.
     *
     * NOTE: This routine does not presume to know what values to set for
     * coordinate components above MILHOJA_NDIM.  Therefore, calling code is
     * responsible for setting or ignoring such data.  This routine will not
     * alter or overwrite such values in the given arrays.
     *
     * \todo Can we put into the C++ code a routine that just takes the
     * pointers?  Ideally those pointers could go all the way to the grid
     * backend and the backend could set the values into the Fortran array
     * directly in one go.  This would be premature optimization at the moment.
     *
     * \param lo      Where to store coordinates of the low point used to define the box
     * \param hi      Where to store coordinates of the high point used to define the box
     * \return The milhoja error code
     */
    int    milhoja_grid_domain_bound_box_c(milhoja::Real* lo, milhoja::Real* hi) {
        using namespace milhoja;

        if (!lo || !hi) {
            std::cerr << "[milhoja_grid_domain_bound_box_c] Invalid pointers" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            Grid&   grid = Grid::instance();
            RealVect probLo = grid.getProbLo();
            RealVect probHi = grid.getProbHi();
            for (unsigned int i=0; i<MILHOJA_NDIM; ++i) {
                lo[i] = probLo[i];
                hi[i] = probHi[i];
            }
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_BOUNDS;
        } catch (...) {
            std::cerr << "[milhoja_grid_domain_bound_box_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_BOUNDS;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the index of the finest mesh refinement level that could be
     * used at any time during execution.
     *
     * \todo error check cast for overflow
     *
     * \param level    The 1-based index of the level where 1 is coarsest
     * \return The milhoja error code
     */
    int    milhoja_grid_max_finest_level_c(int* level) {
        using namespace milhoja;

        if (!level) {
            std::cerr << "[milhoja_grid_max_finest_level_c] Invalid pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            // Level is 0-based in Grid C++ code
            unsigned int   level_ui = Grid::instance().getMaxRefinement() + 1;
            *level = static_cast<int>(level_ui);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        } catch (...) {
            std::cerr << "[milhoja_grid_max_finest_level_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the index of the finest mesh refinement level that is currently
     * in existence and use.
     *
     * \todo error check cast for overflow
     *
     * \param level    The 1-based index of the level where 1 is coarsest
     * \return The milhoja error code
     */
    int    milhoja_grid_current_finest_level_c(int* level) {
        using namespace milhoja;

        if (!level) {
            std::cerr << "[milhoja_grid_current_finest_level_c] Invalid pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            // Level is 0-based in Grid C++ code
            unsigned int   level_ui = Grid::instance().getMaxLevel() + 1;
            *level = static_cast<int>(level_ui);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        } catch (...) {
            std::cerr << "[milhoja_grid_current_finest_level_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the mesh refinement values for the given refinement level.
     *
     * NOTE: This routine does not presume to know what values to set for
     * resolution values above MILHOJA_NDIM.  Therefore, calling code is responsible
     * for setting or ignoring such data.  This routine will not alter or
     * overwrite such values in the given array.
     *
     * \todo Can we put into the C++ code a routine that just takes the
     * pointer?  Ideally that pointer could go all the way to the grid
     * backend and the backend could set the values into the Fortran array
     * directly in one go.  This would be premature optimization at the moment.
     *
     * \param level   The 1-based index of the level of interest with 1
     *                being the coarsest level
     * \param deltas  The mesh resolution values
     * \return The milhoja error code
     */
    int    milhoja_grid_deltas_c(const int level, milhoja::Real* deltas) {
        using namespace milhoja;

        if (!deltas) {
            std::cerr << "[milhoja_grid_deltas_c] Invalid pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        // Level is 0-based in Grid C++ code.  Translate before casting to
        // unsigned int.
        const int    level_cpp = level - 1;
        if (level_cpp < 0) {
            std::cerr << "[milhoja_grid_deltas_c] Invalid level value " << level
                      << std::endl;
            return MILHOJA_ERROR_INVALID_LEVEL;
        }
        const unsigned int    level_cpp_ui = static_cast<unsigned int>(level_cpp);

        try {
            RealVect  deltasVect = Grid::instance().getDeltas(level_cpp_ui);
            for (unsigned int i=0; i<MILHOJA_NDIM; ++i) {
                deltas[i] = deltasVect[i];
            }
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_DELTAS;
        } catch (...) {
            std::cerr << "[milhoja_grid_deltas_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_DELTAS;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Initialize the domain and set the initial conditions such that the mesh
     * refinement across the domain is consistent with the initial conditions.
     *
     * This routine applies the initial conditions within each MPI process on a
     * per-tile basis *without* using the runtime.
     *
     * \param initBlock    Procedure to use to compute and store the initial
     *                     conditions on a single tile
     * \return The milhoja error code
     */
    int    milhoja_grid_init_domain_no_runtime_c(milhoja::INIT_BLOCK_NO_RUNTIME initBlock) {
        try {
            milhoja::Grid::instance().initDomain(initBlock);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        } catch (...) {
            std::cerr << "[milhoja_grid_init_domain_no_runtime_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Initialize the domain and set the initial conditions such that the mesh
     * refinement across the domain is consistent with the initial conditions.
     *
     * This routine applies the initial conditions within each MPI process using
     * the runtime with the CPU-only thread team configuration.
     *
     * \param initBlock    Procedure to use to compute and store the initial
     *                     conditions on a single tile
     * \param tileWrapper  WRITE THIS
     * \param nThreads     Number of threads to be activated in CPU thread team
     * \return The milhoja error code
     */
    int    milhoja_grid_init_domain_cpu_only_c(milhoja::ACTION_ROUTINE initBlock,
                                               void* tileWrapper,
                                               const int nThreads) {
        if (nThreads < 0) {
            std::cerr << "[milhoja_grid_init_domain_cpu_only_c] nThreads is negative" << std::endl;
            return MILHOJA_ERROR_N_THREADS_NEGATIVE;
        }
        unsigned int    nThreads_ui = static_cast<unsigned int>(nThreads);

        milhoja::RuntimeAction     action;
        action.name            = "Init block - C-interface layer";
        action.nInitialThreads = nThreads_ui;
        action.teamType        = milhoja::ThreadTeamDataType::BLOCK;
        action.nTilesPerPacket = 0;
        action.routine         = initBlock;

        milhoja::TileWrapper*   prototype = static_cast<milhoja::TileWrapper*>(tileWrapper);

        try {
            milhoja::Grid::instance().initDomain(action, prototype);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        } catch (...) {
            std::cerr << "[milhoja_grid_init_domain_cpu_only_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the size of all blocks in the domain in terms of number of cells
     * along each edge.
     *
     * NOTE: This function does not presume to know what values to set for
     * dimensions above MILHOJA_NDIM.  Therefore, calling code is responsible
     * for setting or ignoring such data.  This function will not alter or
     * overwrite such variables.
     *
     * \todo Can we put into the C++ code a routine that just takes the
     * pointer?  Ideally that pointer could go all the way to the grid
     * backend and the backend could set the values into the Fortran variables
     * directly in one go.  This would be premature optimization at the moment.
     * \todo Check that n[xyz]b_ui don't have values so large that they overflow
     * when cast to int.
     *
     * \param nxb   The variable whose value is set to the number of cells in
     *              the block along the x axis.
     * \param nyb   Along the y axis.
     * \param nzb   Along the z axis.
     * \return The milhoja error code
     */
    int    milhoja_grid_block_size_c(int* nxb, int* nyb, int* nzb) {
        using namespace milhoja;

        try {
            unsigned int   nxb_ui = 0;
            unsigned int   nyb_ui = 0;
            unsigned int   nzb_ui = 0;

            Grid::instance().getBlockSize(&nxb_ui, &nyb_ui, &nzb_ui);

            if (!nxb) {
                std::cerr << "[milhoja_grid_block_size_c] nxb null" << std::endl;
                return MILHOJA_ERROR_POINTER_IS_NULL;
            }
            *nxb = static_cast<int>(nxb_ui);
#if MILHOJA_NDIM >= 2
            if (!nyb) {
                std::cerr << "[milhoja_grid_block_size_c] nyb null" << std::endl;
                return MILHOJA_ERROR_POINTER_IS_NULL;
            }
            *nyb = static_cast<int>(nyb_ui);
#endif
#if MILHOJA_NDIM == 3
            if (!nzb) {
                std::cerr << "[milhoja_grid_block_size_c] nzb null" << std::endl;
                return MILHOJA_ERROR_POINTER_IS_NULL;
            }
            *nzb = static_cast<int>(nzb_ui);
#endif
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_BLOCK_SIZE;
        } catch (...) {
            std::cerr << "[milhoja_grid_block_size_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_BLOCK_SIZE;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the block domain decomposition of the coarsest level.
     *
     * NOTE: This function does not presume to know what values to set for
     * dimensions above MILHOJA_NDIM.  Therefore, calling code is responsible
     * for setting or ignoring such data.  This function will not alter or
     * overwrite such variables.
     *
     * \todo Can we put into the C++ code a routine that just takes
     * pointers?  Ideally the pointers could go all the way to the grid
     * backend and the backend could set the values into the Fortran variables
     * directly in one go.  This would be premature optimization at the moment.
     * \todo Check that nBlocks[XYZ]_ui don't have values so large that they
     * overflow when cast to int.
     *
     * \param nBlocksX   The pointer whose variable is set to the number of blocks
     *                   along the x axis of the coarsest level.
     * \param nBlocksY   Along the y axis.
     * \param nBlocksZ   Along the z axis.
     * \return The milhoja error code
     */
    int    milhoja_grid_domain_decomposition_c(int* nBlocksX,
                                               int* nBlocksY,
                                               int* nBlocksZ) {
        using namespace milhoja;

        if (!nBlocksX || !nBlocksY || !nBlocksZ) {
            std::cerr << "[milhoja_grid_domain_decomposition_c] Invalid pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            unsigned int   nBlocksX_ui = 0;
            unsigned int   nBlocksY_ui = 0;
            unsigned int   nBlocksZ_ui = 0;

            Grid::instance().getDomainDecomposition(&nBlocksX_ui,
                                                    &nBlocksY_ui,
                                                    &nBlocksZ_ui);

            if (!nBlocksX) {
                std::cerr << "[milhoja_grid_domain_decomposition_c] nBlocksX null" << std::endl;
                return MILHOJA_ERROR_POINTER_IS_NULL;
            }
            *nBlocksX = static_cast<int>(nBlocksX_ui);
#if MILHOJA_NDIM >= 2
            if (!nBlocksY) {
                std::cerr << "[milhoja_grid_domain_decomposition_c] nBlocksY null" << std::endl;
                return MILHOJA_ERROR_POINTER_IS_NULL;
            }
            *nBlocksY = static_cast<int>(nBlocksY_ui);
#endif
#if MILHOJA_NDIM == 3
            if (!nBlocksZ) {
                std::cerr << "[milhoja_grid_domain_decomposition_c] nBlocksZ null" << std::endl;
                return MILHOJA_ERROR_POINTER_IS_NULL;
            }
            *nBlocksZ = static_cast<int>(nBlocksZ_ui);
#endif
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_DOMAIN_DECOMPOSITION;
        } catch (...) {
            std::cerr << "[milhoja_grid_domain_decomposition_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_DOMAIN_DECOMPOSITION;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the number of guardcells for the blocks.
     *
     * \todo Can we put into the C++ code a routine that just takes the
     * pointer?  Ideally that pointer could go all the way to the grid
     * backend and the backend could set the value into the Fortran variable
     * directly in one go.  This would be premature optimization at the moment.
     * \todo Check that nGuardcells_ui doesn't have a value so large that it
     * overflows when cast to int.
     *
     * \param nGuardcells   The variable whose value is set to the number of
     *                      guardcells
     * \return The milhoja error code
     */
    int    milhoja_grid_n_guardcells_c(int* nGuardcells) {
        using namespace milhoja;

        if (!nGuardcells) {
            std::cerr << "[milhoja_grid_n_guardcells_c] Invalid pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            unsigned int   nGuardcells_ui = 0;

            nGuardcells_ui = Grid::instance().getNGuardcells();

            *nGuardcells = static_cast<int>(nGuardcells_ui);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_N_GUARDCELLS;
        } catch (...) {
            std::cerr << "[milhoja_grid_n_guardcells_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_N_GUARDCELLS;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the number of cell-centered variables stored in each block.
     *
     * \todo Can we put into the C++ code a routine that just takes the
     * pointer?  Ideally that pointer could go all the way to the grid
     * backend and the backend could set the value into the Fortran variable
     * directly in one go.  This would be premature optimization at the moment.
     * \todo Check that nCcVars_ui doesn't have a value so large that it
     * overflows when cast to int.
     *
     * \param nCcVars   The variable whose value is set to the number of
     *                  variables
     * \return The milhoja error code
     */
    int    milhoja_grid_n_cc_variables_c(int* nCcVars) {
        using namespace milhoja;

        if (!nCcVars) {
            std::cerr << "[milhoja_grid_n_cc_variables_c] Invalid pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            unsigned int   nCcVars_ui = 0;

            nCcVars_ui = Grid::instance().getNCcVariables();

            *nCcVars = static_cast<int>(nCcVars_ui);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_N_CC_VARS;
        } catch (...) {
            std::cerr << "[milhoja_grid_n_cc_variables_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_N_CC_VARS;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * Obtain the number of flux variables.
     *
     * \todo Can we put into the C++ code a routine that just takes the
     * pointer?  Ideally that pointer could go all the way to the grid
     * backend and the backend could set the value into the Fortran variable
     * directly in one go.  This would be premature optimization at the moment.
     * \todo Check that nFluxVars_ui doesn't have a value so large that it
     * overflows when cast to int.
     *
     * \param nFluxVars   The variable whose value is set to the number of
     *                    variables
     * \return The milhoja error code
     */
    int    milhoja_grid_n_flux_variables_c(int* nFluxVars) {
        if (!nFluxVars) {
            std::cerr << "[milhoja_grid_n_flux_variables_c] Invalid pointer" << std::endl;
            return MILHOJA_ERROR_POINTER_IS_NULL;
        }

        try {
            unsigned int   nFluxVars_ui = 0;

            nFluxVars_ui = milhoja::Grid::instance().getNFluxVariables();

            *nFluxVars = static_cast<int>(nFluxVars_ui);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_N_FLUX_VARS;
        } catch (...) {
            std::cerr << "[milhoja_grid_n_flux_variables_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_N_FLUX_VARS;
        }

        return MILHOJA_SUCCESS;
    }
#endif

#ifdef FULL_MILHOJAGRID
    /**
     * \todo Allow calling code to specify filename.  No need for step in that
     *       case.
     */
    int   milhoja_grid_write_plotfile_c(const int step) {
        if (step < 0) {
            std::cerr << "[milhoja_grid_write_plotfile_c] Step is negative" << std::endl;
            return MILHOJA_ERROR_STEP_NEGATIVE;
        }

        try {
            std::string    filename = "milhoja_plt_" + std::to_string(step);

            milhoja::Grid&    grid = milhoja::Grid::instance();

            std::vector<std::string>   names{grid.getNCcVariables()};
            for (auto n=0; n<names.size(); ++n) {
                names[n] = "var" + std::to_string(n + 1);
            }

            grid.writePlotfile(filename, names);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_WRITE_PLOTFILE;
        } catch (...) {
            std::cerr << "[milhoja_grid_write_plotfile_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_WRITE_PLOTFILE;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     *
     */
     int    milhoja_grid_fill_guardcells_c(void) {
        try {
            milhoja::Grid::instance().fillGuardCells();
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FILL_GCS;
        } catch (...) {
            std::cerr << "[milhoja_grid_fill_guardcells_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FILL_GCS;
        }

        return MILHOJA_SUCCESS;
     }
#endif
}

