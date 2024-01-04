#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>
#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>
#include <Milhoja_test.h>

#include "Base.h"
#include "RuntimeParameters.h"
#include "setInitialInteriorTest.h"
#include "errorEstMultiple.h"

int main(int argc, char* argv[]) {
    using namespace milhoja;

    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    int     exitCode = 1;
    try {
        Logger::initialize("GridMultipleUnitTest.log", GLOBAL_COMM, LEAD_RANK);
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
            cfg.errorEstimation         = Simulation::errorEstMultiple;
            cfg.mpiComm                 = GLOBAL_COMM;

            cfg.load();
        }
        Grid::initialize();
        Grid::instance().initDomain(Simulation::setInitialInteriorTest);

        exitCode = RUN_ALL_TESTS();

        Grid::instance().destroyDomain();
        Grid::instance().finalize();

        RuntimeParameters::instance().finalize();
        Logger::instance().finalize();
    } catch(const std::exception& e) {
        std::cerr << "FAILURE - Grid/multiple::main - " << e.what() << std::endl;
        exitCode = 111;
    } catch(...) {
        std::cerr << "FAILURE - Grid/multiple::main - Exception of unexpected type caught"
                  << std::endl;
        exitCode = 222;
    }

    MPI_Finalize();

    return exitCode;
}

