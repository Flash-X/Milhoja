#include <mpi.h>

#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Runtime.h>
#include <Milhoja_Logger.h>

#include "Sedov.h"
#include "RuntimeParameters.h"
#include "Io.h"
#include "Driver.h"
#include "errorEstBlank.h"

int main(int argc, char* argv[]) {
    constexpr int    LOG_RANK = LEAD_RANK;
    constexpr int    IO_RANK  = LEAD_RANK;

    MPI_Init(&argc, &argv);

    int     exitCode = 0;
    try {
        milhoja::Logger::initialize("sedov.log", GLOBAL_COMM, LOG_RANK);

        RuntimeParameters::initialize("RuntimeParameters.json");
        RuntimeParameters&    RPs = RuntimeParameters::instance();

        milhoja::Runtime::initialize(
                    RPs.getUnsignedInt("Runtime", "nThreadTeams"),
                    RPs.getUnsignedInt("Runtime", "nThreadsPerTeam"),
                    RPs.getUnsignedInt("Runtime", "nStreams"),
                    RPs.getSizeT("Runtime", "memoryPoolSizeBytes"));

        // Configure in local block so that code cannot accidentally access
        // configuration data after it is consumed by Grid at initialization.
        {
            milhoja::GridConfiguration&   cfg = milhoja::GridConfiguration::instance();

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
            cfg.ccInterpolator          = milhoja::Interpolator::CellConservativeLinear;
            cfg.mpiComm                 = GLOBAL_COMM;

            cfg.load();
        }
        milhoja::Grid::initialize();

        Io::instantiate("sedov.dat", GLOBAL_COMM, IO_RANK);

        // All allocation of simulation-specific resources occurs in the local
        // scope of this function.  They are therefore released/destroyed before
        // we do high-level clean-up next, which is a good practice.
        Driver::executeSimulation();

        milhoja::Grid::instance().finalize();
        milhoja::Runtime::instance().finalize();
        RuntimeParameters::instance().finalize();
        milhoja::Logger::instance().finalize();
    } catch(const std::exception& e) {
        std::cerr << "FAILURE - " << e.what() << std::endl;
        exitCode = 111;
    } catch(...) {
        std::cerr << "FAILURE - Exception of unexpected type caught" << std::endl;
        exitCode = 222;
    }

    MPI_Finalize();

    return exitCode;
}

