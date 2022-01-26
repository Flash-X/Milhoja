#include <stdexcept>

#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>
#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>

#include "Base.h"
#include "setInitialConditions.h"
#include "errorEstMaximal.h"
#include "Flash_par.h"

int main(int argc, char* argv[]) {
    using namespace milhoja;

    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    int     exitCode = 1;
    try {
        Logger::initialize("GridGeneralUnitTest.log", GLOBAL_COMM, LEAD_RANK);

        // Access config singleton within limited local scope so that it can't be
        // used by the rest of the application code outside the block.
        {
            GridConfiguration&   cfg = GridConfiguration::instance();

            cfg.xMin            = rp_Grid::X_MIN;
            cfg.xMax            = rp_Grid::X_MAX;
            cfg.yMin            = rp_Grid::Y_MIN;
            cfg.yMax            = rp_Grid::Y_MAX;
            cfg.zMin            = rp_Grid::Z_MIN;
            cfg.zMax            = rp_Grid::Z_MAX;
            cfg.nxb             = rp_Grid::NXB;
            cfg.nyb             = rp_Grid::NYB;
            cfg.nzb             = rp_Grid::NZB;
            cfg.nCcVars         = NUNKVAR;
            cfg.nGuard          = NGUARD;
            cfg.nBlocksX        = rp_Grid::N_BLOCKS_X;
            cfg.nBlocksY        = rp_Grid::N_BLOCKS_Y;
            cfg.nBlocksZ        = rp_Grid::N_BLOCKS_Z;
            cfg.maxFinestLevel  = rp_Grid::LREFINE_MAX;
            cfg.errorEstimation = Simulation::errorEstMaximal;
            cfg.mpiComm         = GLOBAL_COMM;

            cfg.load();
        }

        // We test throughout logic errors related to the use of the Grid in
        // the high-level application control flow, some of which cannot be
        // included in a googletest.
        try {
            // We cannot finalize without accessing the singleton.  Therefore this
            // also proves that we cannot finalize without first initializing.
            Grid::instance();
            std::cerr << "FAILURE - Grid/general::main - Accessed Grid singleton before init"
                      << std::endl;
            return 1;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        Grid::initialize();

        try {
            Grid::initialize();
            std::cerr << "FAILURE - Grid/general::main - Grid initialized more than once"
                      << std::endl;
            return 2;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        Grid&   grid = Grid::instance();

        try {
            grid.destroyDomain();
            std::cerr << "FAILURE - Grid/general::main - Domain destroyed without init"
                      << std::endl;
            return 3;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu);

        // There is a test that confirms that initDomain can be called at most once.
        exitCode = RUN_ALL_TESTS();

        try {
            grid.finalize();
            std::cerr << "FAILURE - Grid/general::main - Grid finalized without destroying domain"
                      << std::endl;
            return 4;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        grid.destroyDomain();
        grid.finalize();

        try {
            grid.finalize();
            std::cerr << "FAILURE - Grid/general::main - Grid finalized more than once"
                      << std::endl;
            return 5;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        try {
            Grid::instance();
            std::cerr << "FAILURE - Grid/general::main - Grid accessed after finalize"
                      << std::endl;
            return 6;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        Logger::instance().finalize();
    } catch(const std::exception& e) {
        std::cerr << "FAILURE - Grid/general::main - " << e.what() << std::endl;
        exitCode = 111;
    } catch(...) {
        std::cerr << "FAILURE - Grid/general::main - Exception of unexpected type caught"
                  << std::endl;
        exitCode = 222;
    }

    MPI_Finalize();

    return exitCode;
}

