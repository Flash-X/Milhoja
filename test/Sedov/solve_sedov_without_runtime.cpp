#include <mpi.h>

#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Logger.h>

#include "Sedov.h"
#include "Io.h"
#include "Driver.h"
#include "errorEstBlank.h"

#include "Flash_par.h"

int main(int argc, char* argv[]) {
    constexpr int    LOG_RANK = LEAD_RANK;
    constexpr int    IO_RANK  = LEAD_RANK;

    MPI_Init(&argc, &argv);

    int     exitCode = 0;
    try {
        milhoja::Logger::initialize(rp_Simulation::LOG_FILENAME,
                                    GLOBAL_COMM, LOG_RANK);

        // Configure in local block so that code cannot accidentally access
        // configuration data after it is consumed by Grid at initialization.
        {
            milhoja::GridConfiguration&   cfg = milhoja::GridConfiguration::instance();

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
            cfg.errorEstimation = Simulation::errorEstBlank;
            cfg.mpiComm         = GLOBAL_COMM;

            cfg.load();
        }
        milhoja::Grid::initialize();

        Io::instantiate(rp_Simulation::INTEGRAL_QUANTITIES_FILENAME,
                        GLOBAL_COMM, IO_RANK);

        // All allocation of simulation-specific resources occurs in the local
        // scope of this function.  They are therefore released/destroyed before
        // we do high-level clean-up next, which is a good practice.
        Driver::executeSimulation();

        milhoja::Grid::instance().finalize();
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

