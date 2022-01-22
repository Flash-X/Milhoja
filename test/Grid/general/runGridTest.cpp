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

    Logger::instantiate("GridGeneralUnitTest.log", GLOBAL_COMM, LEAD_RANK);

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

    // We test here logic errors that cannot be included in a googletest because
    // those all run with the singleton and domain initialized.
    //
    // Must initialize Grid singleton before accessing it.  We cannot finalize
    // without getting the singleton.  Therefore this also proves that we cannot
    // finalize without first initializing.
    try {
        Grid::instance();
        std::cerr << "Expected exception not thrown" << std::endl;
        return 1;
    } catch(const std::logic_error& e) {
        // Ignore since this is this expected behavior.
    } catch(...) {
        std::cerr << "Unexpected exception caught" << std::endl;
        return 2;
    }

    Grid::initialize();

    // Cannot initialize the singleton more than once
    try {
        Grid::initialize();
        std::cerr << "Expected exception not thrown" << std::endl;
        return 3;
    } catch(const std::logic_error& e) {
        // Ignore since this is this expected behavior.
    } catch(...) {
        std::cerr << "Unexpected exception caught" << std::endl;
        return 4;
    }

    Grid&   grid = Grid::instance();

    // Cannot destroy domain that hasn't been initialized
    try {
        grid.destroyDomain();
        std::cerr << "Expected exception not thrown" << std::endl;
        return 5;
    } catch(const std::logic_error& e) {
        // Ignore since this is this expected behavior.
    } catch(...) {
        std::cerr << "Unexpected exception caught" << std::endl;
        return 6;
    }

    grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu);

    // There is a test that confirms that initDomain can be called at most once.
    int exitCode = RUN_ALL_TESTS();

    // destroyDomain must be called before calling finalize
    try {
        grid.finalize();
        std::cerr << "Expected exception not thrown" << std::endl;
        return 7;
    } catch(const std::logic_error& e) {
        // Ignore since this is this expected behavior.
    } catch(...) {
        std::cerr << "Unexpected exception caught" << std::endl;
        return 8;
    }

    grid.destroyDomain();
    grid.finalize();

    // Cannot finalize more than once
    try {
        grid.finalize();
        std::cerr << "Expected exception not thrown" << std::endl;
        return 9;
    } catch(const std::logic_error& e) {
        // Ignore since this is this expected behavior.
    } catch(...) {
        std::cerr << "Unexpected exception caught" << std::endl;
        return 10;
    }

    // Cannot access singleton after finalizing
    try {
        Grid::instance();
        std::cerr << "Expected exception not thrown" << std::endl;
        return 11;
    } catch(const std::logic_error& e) {
        // Ignore since this is this expected behavior.
    } catch(...) {
        std::cerr << "Unexpected exception caught" << std::endl;
        return 12;
    }

    MPI_Finalize();

    return exitCode;
}

