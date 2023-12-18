#include <stdexcept>

#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>
#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>
#include <Milhoja_test.h>

#include "Base.h"
#include "RuntimeParameters.h"
#include "setInitialConditions.h"
#include "errorEstMaximal.h"

int main(int argc, char* argv[]) {
    using namespace milhoja;

    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    int     exitCode = 1;
    try {
        Logger::initialize("GridGeneralUnitTest.log", GLOBAL_COMM, LEAD_RANK);
        RuntimeParameters::initialize("RuntimeParameters.json");

        // Access config singleton within limited local scope so that it can't be
        // used by the rest of the application code outside the block.
        {
            GridConfiguration&   cfg = GridConfiguration::instance();
            RuntimeParameters&   RPs = RuntimeParameters::instance();

            cfg.xMin                    = RPs.getReal("Grid", "xMin");
            cfg.xMax                    = RPs.getReal("Grid", "xMax");
            cfg.yMin                    = RPs.getReal("Grid", "yMin");
            cfg.yMax                    = RPs.getReal("Grid", "yMax");
            cfg.zMin                    = RPs.getReal("Grid", "zMin");
            cfg.zMax                    = RPs.getReal("Grid", "zMax");
            cfg.nxb                     = MILHOJA_TEST_NXB;
            cfg.nyb                     = MILHOJA_TEST_NYB;
            cfg.nzb                     = MILHOJA_TEST_NZB;
            cfg.nCcVars                 = NUNKVAR;
            cfg.nFluxVars               = NFLUXES;
            cfg.loBCs[milhoja::Axis::I] = milhoja::BCs::Periodic;
            cfg.hiBCs[milhoja::Axis::I] = milhoja::BCs::Periodic;
            cfg.loBCs[milhoja::Axis::J] = milhoja::BCs::Periodic;
            cfg.hiBCs[milhoja::Axis::J] = milhoja::BCs::Periodic;
            cfg.loBCs[milhoja::Axis::K] = milhoja::BCs::Periodic;
            cfg.hiBCs[milhoja::Axis::K] = milhoja::BCs::Periodic;
            cfg.externalBcRoutine       = nullptr;
            cfg.nGuard                  = NGUARD;
            cfg.nBlocksX                = RPs.getUnsignedInt("Grid", "nBlocksX");
            cfg.nBlocksY                = RPs.getUnsignedInt("Grid", "nBlocksY");
            cfg.nBlocksZ                = RPs.getUnsignedInt("Grid", "nBlocksZ");
            cfg.maxFinestLevel          = RPs.getUnsignedInt("Grid", "finestRefinementLevel");
            cfg.errorEstimation         = Simulation::errorEstMaximal;
            cfg.mpiComm                 = GLOBAL_COMM;

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

        grid.initDomain(sim::setInitialConditions_noRuntime);

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

        RuntimeParameters::instance().finalize();
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

