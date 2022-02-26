/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in Milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <iostream>

#include <mpi.h>

#include "Milhoja.h"
#include "Milhoja_Logger.h"
#include "Milhoja_GridConfiguration.h"
#include "Milhoja_Grid.h"
#include "Milhoja_interface_error_codes.h"
#include "Milhoja_actionRoutine.h"

extern "C" {
    /**
     * Perform all Milhoja initializations needed for calling code to begin
     * using its grid infrastructure.  Note that calling code must subsequently
     * call Grid_initDomain when deemed appropriate to setup the problem domain
     * in the grid infrastructure.  Calling code must initialize MPI before
     * calling this routine.
     *
     * \todo Milhoja_GridConfigurationAMReX presently expects
     * maxRefinementLevel to be 1-based.  Is this really what the C++ code wants
     * or should this function convert to 0-based?
     *
     * \param globalCommF          The Fortran version of the MPI communicator that
     *                             Milhoja should use
     * \param logRank              The rank in the given communicator of the MPI process
     *                             that should perform logging duties.
     * \param xMin                 Define the physical domain in X as [xMin, xMax]
     * \param xMax                 See xMin
     * \param yMin                 Define the physical domain in Y as [yMin, yMax]
     * \param yMax                 See yMin
     * \param zMin                 Define the physical domain in Z as [zMin, zMax]
     * \param zMax                 See zMin
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
     * \param nGuard               The number of guardcells
     * \param nCcVars              The number of physical variables in the solution
     * \param errorEst             Procedure that is used to assess if a block should
     *                             be refined, derefined, or stay at the same
     *                             refinement
     * \return The milhoja error code
     */
    int    milhoja_grid_init_c(const MPI_Fint globalCommF,
                               const int logRank,
                               const milhoja::Real xMin, const milhoja::Real xMax,
                               const milhoja::Real yMin, const milhoja::Real yMax,
                               const milhoja::Real zMin, const milhoja::Real zMax,
                               const int nxb, const int nyb, const int nzb,
                               const int nBlocksX,
                               const int nBlocksY,
                               const int nBlocksZ,
                               const int maxRefinementLevel,
                               const int nGuard, const int nCcVars,
                               milhoja::ERROR_ROUTINE errorEst) {
        MPI_Comm    globalComm = MPI_Comm_f2c(globalCommF);

        if (   (nxb      < 0) || (nyb      < 0) || (nzb      < 0) 
            || (nBlocksX < 0) || (nBlocksY < 0) || (nBlocksZ < 0)
            || (maxRefinementLevel < 0) || (nGuard < 0) || (nCcVars < 0)) {
            std::cerr << "[milhoja_grid_init_c] Invalid configuration value" << std::endl;
            return MILHOJA_ERROR_NEGATIVE_VALUE_FOR_UINT;
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

        try {
            milhoja::Logger::initialize("milhoja.log", globalComm, logRank);

            // Configure in local block so that code cannot accidentally access
            // configuration data after it is consumed by Grid at initialization.
            milhoja::GridConfiguration&   cfg = milhoja::GridConfiguration::instance();
   
            cfg.xMin            = xMin;
            cfg.xMax            = xMax;
            cfg.yMin            = yMin;
            cfg.yMax            = yMax;
            cfg.zMin            = zMin;
            cfg.zMax            = zMax;
            cfg.nxb             = nxb_ui;
            cfg.nyb             = nyb_ui;
            cfg.nzb             = nzb_ui;
            cfg.nCcVars         = nCcVars_ui;
            cfg.nGuard          = nGuard_ui;
            cfg.nBlocksX        = nBlocksX_ui;
            cfg.nBlocksY        = nBlocksY_ui;
            cfg.nBlocksZ        = nBlocksZ_ui;
            cfg.maxFinestLevel  = maxRefinementLevel_ui;
            cfg.errorEstimation = errorEst;
            cfg.mpiComm         = globalComm;
   
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
     * Calling code should finalize the grid before finalizing the runtime.
     *
     * \return The milhoja error code
     */
    int    milhoja_grid_finalize_c(void) {
        try {
            milhoja::Grid::instance().destroyDomain();
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

    /**
     * Obtain the low and high coordinates in physical space of the rectangular
     * box that bounds the problem's spatial domain.
     *
     * NOTE: This routine does not presume to know what values to set for
     * coordinate components above MILHOJA_NDIM.  Therefore, calling code is responsible
     * for setting or ignoring such data.  This routine will not alter or
     * overwrite such values in the given arrays.
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
            std::cerr << "[milhoja_grid_max_finest_level_c] Invalid pointers" << std::endl;
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
            std::cerr << "[milhoja_grid_current_finest_level_c] Invalid pointers" << std::endl;
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
     * \param level   The 1-based index of the level of interest with 1
     *                being the coarsest level
     * \param deltas  The mesh resolution values
     * \return The milhoja error code
     */
    int    milhoja_grid_deltas_c(const int level, milhoja::Real* deltas) {
        using namespace milhoja;

        if (!deltas) {
            std::cerr << "[milhoja_grid_deltas_c] Invalid pointers" << std::endl;
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
     * per-tile basis, which implies that it does *not* use the runtime.
     *
     * \param initBlock    Procedure to use to compute and store the initial
     *                     conditions on a single tile
     * \return The milhoja error code
     */
    int    milhoja_grid_init_domain_c(milhoja::ACTION_ROUTINE initBlock) {
        try {
            milhoja::Grid::instance().initDomain(initBlock);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        } catch (...) {
            std::cerr << "[milhoja_grid_init_domain_c] Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        }

        return MILHOJA_SUCCESS;
    }
}

