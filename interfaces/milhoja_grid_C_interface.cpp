/**
 * C/C++ interoperability layer - compile C++ code with C linkage convention
 *
 * Refer to documentation in milhoja_runtime_C_interface for general
 * Fortran/C/C++ interoperability documentation.
 */

#include <iostream>

#include <mpi.h>

#include "milhoja_interface_error_codes.h"
#include "actionRoutine.h"
#include "Grid.h"

extern "C" {
    /**
     * \todo Error check given communicator and MPI_Comm_f2c call.
     * \todo Error check all casts.
     * \todo Rename lRefineMax to something clearer like maxRefinementLevel.
     */
    int    milhoja_grid_init_c(const int globalCommF,
                               const orchestration::Real xMin, const orchestration::Real xMax,
                               const orchestration::Real yMin, const orchestration::Real yMax,
                               const orchestration::Real zMin, const orchestration::Real zMax,
                               const int nxb, const int nyb, const int nzb,
                               const int nBlocksX,
                               const int nBlocksY,
                               const int nBlocksZ,
                               const int lRefineMax,
                               const int nGuard, const int nCcVars,
                               orchestration::ACTION_ROUTINE initBlock,
                               orchestration::ERROR_ROUTINE errorEst) {
        MPI_Comm    globalComm = MPI_Comm_f2c(globalCommF);

        unsigned int    nxb_ui        = static_cast<unsigned int>(nxb);
        unsigned int    nyb_ui        = static_cast<unsigned int>(nyb);
        unsigned int    nzb_ui        = static_cast<unsigned int>(nzb);
        unsigned int    nBlocksX_ui   = static_cast<unsigned int>(nBlocksX);
        unsigned int    nBlocksY_ui   = static_cast<unsigned int>(nBlocksY);
        unsigned int    nBlocksZ_ui   = static_cast<unsigned int>(nBlocksZ);
        unsigned int    lRefineMax_ui = static_cast<unsigned int>(lRefineMax);
        unsigned int    nGuard_ui     = static_cast<unsigned int>(nGuard);
        unsigned int    nCcVars_ui    = static_cast<unsigned int>(nCcVars);

        try {
            orchestration::Grid::instantiate(globalComm,
                                             xMin, xMax,
                                             yMin, yMax,
                                             zMin, zMax,
                                             nxb_ui, nyb_ui, nzb_ui,
                                             nBlocksX_ui, nBlocksY_ui, nBlocksZ_ui,
                                             lRefineMax_ui,
                                             nGuard_ui, nCcVars_ui,
                                             initBlock, errorEst);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_init_c] Unable to initialize\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_GRID;
        } catch (...) {
            std::cerr << "[milhoja_grid_init_c] Unable to initialize\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_GRID;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     *
     */
    int    milhoja_grid_finalize_c(void) {
        try {
            orchestration::Grid::instance().finalize();
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_finalize_c] Unable to finalize\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FINALIZE_GRID;
        } catch (...) {
            std::cerr << "[milhoja_grid_finalize_c] Unable to finalize\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_FINALIZE_GRID;
        }
        
        return MILHOJA_SUCCESS;
    }

    /**
     *
     */
    int    milhoja_grid_init_domain_c(const int nDistributorThreads,
                                      const int nTeamThreads) {
        if        (nDistributorThreads < 0) {
            std::cerr << "[milhoja_runtime_init_domain_c] nDistributorThreads is negative" << std::endl;
            return MILHOJA_ERROR_N_DISTRIBUTOR_THREADS_NEGATIVE;
        } else if (nTeamThreads < 0) {
            std::cerr << "[milhoja_runtime_init_domain_c] nTeamThreads is negative" << std::endl;
            return MILHOJA_ERROR_N_TEAM_THREADS_NEGATIVE;
        }

        unsigned int   nDistributorThreads_ui = static_cast<unsigned int>(nDistributorThreads);
        unsigned int   nTeamThreads_ui        = static_cast<unsigned int>(nTeamThreads);

        try {
            orchestration::Grid::instance().initDomain(nDistributorThreads_ui,
                                                       nTeamThreads_ui);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_init_domain_c] Unable to initialize\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        } catch (...) {
            std::cerr << "[milhoja_grid_init_domain_c] Unable to initialize\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * \todo error check given pointers
     */
    int    milhoja_grid_domain_bound_box_c(orchestration::Real* lo,
                                           orchestration::Real* hi) {
        using namespace orchestration;

        try {
            Grid&   grid = Grid::instance();
            RealVect probLo = grid.getProbLo();
            RealVect probHi = grid.getProbHi();
            for (unsigned int i=0; i<NDIM; ++i) {
                lo[i] = probLo[i];
                hi[i] = probHi[i];
            }
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_domain_bound_box_c] Unable to get bounds\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_BOUNDS;
        } catch (...) {
            std::cerr << "[milhoja_grid_domain_bound_box_c] Unable to get bounds\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_BOUNDS;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * level index is 1-based
     *
     * \todo error check given pointer
     */
    int    milhoja_grid_deltas_c(const int level, orchestration::Real* deltas) {
        using namespace orchestration;

        if (level < 0) {
            std::cerr << "[milhoja_grid_deltas_c] level is negative" << std::endl;
            return MILHOJA_ERROR_LEVEL_NEGATIVE;
        }
        unsigned int  level_ui = static_cast<unsigned int>(level);

        try {
            // Level is 0-based in Grid C++ code
            RealVect  deltasVect = Grid::instance().getDeltas(level_ui - 1);
            for (unsigned int i=0; i<NDIM; ++i) {
                deltas[i] = deltasVect[i];
            }
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_deltas_c] Unable to get deltas\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_DELTAS;
        } catch (...) {
            std::cerr << "[milhoja_grid_deltas_c] Unable to get deltas\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_DELTAS;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * level index is 1-based
     *
     * \todo error check given pointer
     * \todo error check cast for overflow
     */
    int    milhoja_grid_max_finest_level_c(int* level) {
        using namespace orchestration;

        try {
            // Level is 0-based in Grid C++ code
            unsigned int   level_ui = Grid::instance().getMaxRefinement() + 1;
            *level = static_cast<int>(level_ui);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_max_finest_level_c] Unable to get level\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        } catch (...) {
            std::cerr << "[milhoja_grid_max_finest_level_c] Unable to get level\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * level index is 1-based
     *
     * \todo error check given pointer
     * \todo error check cast for overflow
     */
    int    milhoja_grid_current_finest_level_c(int* level) {
        using namespace orchestration;

        try {
            // Level is 0-based in Grid C++ code
            unsigned int   level_ui = Grid::instance().getMaxLevel() + 1;
            *level = static_cast<int>(level_ui);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_current_finest_level_c] Unable to get level\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        } catch (...) {
            std::cerr << "[milhoja_grid_current_finest_level_c] Unable to get level\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_GET_LEVEL;
        }

        return MILHOJA_SUCCESS;
    }

    /**
     * \todo Allow calling code to specify filename.  No need for step in that
     *       case.
     */
    int   milhoja_grid_write_plotfile_c(const int step) {
        if (step < 0) {
            std::cerr << "[milhoja_grid_write_plotfile_c] Step is negative" << std::endl;
            return MILHOJA_ERROR_STEP_NEGATIVE;
        }

        std::string   filename = "milhoja_plt_" + std::to_string(step);

        try {
            orchestration::Grid::instance().writePlotfile(filename);
        } catch (const std::exception& exc) {
            std::cerr << "[milhoja_grid_write_plotfile_c] Unable to write plot\n" 
                      << exc.what() << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_WRITE_PLOTFILE;
        } catch (...) {
            std::cerr << "[milhoja_grid_write_plotfile_c] Unable to write plot\n" 
                      << "Unknown error caught" << std::endl;
            return MILHOJA_ERROR_UNABLE_TO_WRITE_PLOTFILE;
        }

        return MILHOJA_SUCCESS;
    }
}

