#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>
#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Runtime.h>

#include "Base.h"
#include "RuntimeParameters.h"
#include "setInitialConditions.h"
#include "errorEstBlank.h"

int main(int argc, char* argv[]) {
    using namespace milhoja;

    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // We need to create our own main for the testsuite since we can only call
    // MPI_Init/MPI_Finalize once per testsuite execution.
    MPI_Init(&argc, &argv);

    int  rank = -1;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    int     exitCode = 1;
    try {
        Logger::initialize("RuntimeTest.log", GLOBAL_COMM, LEAD_RANK);
        RuntimeParameters::initialize("RuntimeParameters.json");

        RuntimeParameters&   RPs = RuntimeParameters::instance();

        // We test throughout logic errors related to the use of the runtime in
        // the high-level application control flow, some of which cannot be
        // included in a googletest.
        try {
            // We cannot finalize without getting the singleton.  Therefore this
            // also proves that we cannot finalize without first initializing.
            Runtime::instance();
            std::cerr << "FAILURE - Runtime::main - Accessed runtime before init"
                      << std::endl;
            return 1;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        Runtime::initialize(RPs.getUnsignedInt("Runtime", "nThreadTeams"),
                            RPs.getUnsignedInt("Runtime", "nThreadsPerTeam"),
                            RPs.getUnsignedInt("Runtime", "nStreams"),
                            RPs.getSizeT("Runtime", "cpuMemoryPoolSizeBytes"),
                            RPs.getSizeT("Runtime", "gpuMemoryPoolSizeBytes"));

        try {
            Runtime::initialize(RPs.getUnsignedInt("Runtime", "nThreadTeams"),
                                RPs.getUnsignedInt("Runtime", "nThreadsPerTeam"),
                                RPs.getUnsignedInt("Runtime", "nStreams"),
                                RPs.getSizeT("Runtime", "cpuMemoryPoolSizeBytes"),
                                RPs.getSizeT("Runtime", "gpuMemoryPoolSizeBytes"));
            std::cerr << "FAILURE - Runtime::main - Runtime initialized more than once"
                      << std::endl;
            return 2;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        Runtime&             runtime = Runtime::instance();

        // Access config singleton within limited local scope so that it can't be
        // used by the rest of the application code outside the block.
        {
            GridConfiguration&   cfg = GridConfiguration::instance();

            cfg.xMin                    = RPs.getReal("Grid", "xMin");
            cfg.xMax                    = RPs.getReal("Grid", "xMax");
            cfg.yMin                    = RPs.getReal("Grid", "yMin");
            cfg.yMax                    = RPs.getReal("Grid", "yMax");
            cfg.zMin                    = RPs.getReal("Grid", "zMin");
            cfg.zMax                    = RPs.getReal("Grid", "zMax");
            cfg.nxb                     = RPs.getUnsignedInt("Grid", "NXB");
            cfg.nyb                     = RPs.getUnsignedInt("Grid", "NYB");
            cfg.nzb                     = RPs.getUnsignedInt("Grid", "NZB");
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
            cfg.errorEstimation         = Simulation::errorEstBlank;
            cfg.mpiComm                 = GLOBAL_COMM;

            cfg.load();
        }
        Grid::initialize();
        Grid&   grid = Grid::instance();

        RuntimeAction   initBlock_cpu;
        initBlock_cpu.name = "initBlock_cpu";
        initBlock_cpu.teamType        = ThreadTeamDataType::BLOCK;
        initBlock_cpu.nInitialThreads = RPs.getUnsignedInt("Simulation", "nThreadsForIC");
        initBlock_cpu.nTilesPerPacket = 0;
        initBlock_cpu.routine         = ActionRoutines::setInitialConditions_tile_cpu;

        grid.initDomain(initBlock_cpu);

        // All allocation of test-specific resources occurs in the local scope
        // of the tests.  They are therefore released/destroyed before we do
        // high-level clean-up next, which is a good practice.
        exitCode = RUN_ALL_TESTS();

        grid.destroyDomain();
        grid.finalize();
        runtime.finalize();

        try {
            runtime.finalize();
            std::cerr << "FAILURE - Runtime::main - Runtime finalized more than once"
                        << std::endl;
            return 3;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        try {
            // This implies that finalization cannot be called more than once.
            Runtime::instance();
            std::cerr << "FAILURE - Runtime::main - Runtime accessed after finalize"
                      << std::endl;
            return 4;
        } catch(const std::logic_error& e) {
            // Ignore since this is the expected behavior.
        } catch(...) {
            throw;
        }

        RuntimeParameters::instance().finalize();
        Logger::instance().finalize();
    } catch(const std::exception& e) {
        std::cerr << "FAILURE - Runtime::main - " << e.what() << std::endl;
        exitCode = 111;
    } catch(...) {
        std::cerr << "FAILURE - Runtime::main - Exception of unexpected type caught"
                  << std::endl;
        exitCode = 222;
    }

    MPI_Finalize();

    return exitCode;
}

