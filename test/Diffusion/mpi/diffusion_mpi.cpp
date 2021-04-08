#include <cstdio>
#include <string>

#include <mpi.h>

#include "HeatAD.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Timer.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

#include "Flash_par.h"

constexpr  unsigned int N_DIST_THREADS      = 0;
constexpr  unsigned int N_CPU_THREADS       = 1;
constexpr  unsigned int N_GPU_THREADS       = 0;
constexpr  unsigned int N_BLKS_PER_PACKET   = 0;
constexpr  unsigned int N_BLKS_PER_CPU_TURN = 1;

int main(int argc, char* argv[]) {
    // TODO: Add in error handling code

    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    orchestration::Logger::instantiate(rp_Simulation::LOG_FILENAME);

    // Analogous to calling Grid_init
    orchestration::Grid::instantiate();

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    ProcessTimer  diffusion{rp_Simulation::NAME + "_timings.dat", "MPI",
                            N_DIST_THREADS, N_CPU_THREADS, N_GPU_THREADS,
                            N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN};

    //----- MIMIC Grid_initDomain
    orchestration::Grid&     grid    = orchestration::Grid::instance();
    orchestration::Logger&   logger  = orchestration::Logger::instance();

    Driver::dt      = rp_Simulation::DT_INIT;
    Driver::simTime = rp_Simulation::T_0;

    orchestration::Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);
    orchestration::Timer::stop("Set initial conditions");

    //----- MIMIC Driver_evolveFlash
    orchestration::Timer::start(rp_Simulation::NAME + " simulation");

    unsigned int            level{0};
    std::shared_ptr<Tile>   tileDesc{};
    unsigned int            nStep{1};
    while ((nStep <= rp_Simulation::MAX_STEPS) && (Driver::simTime < rp_Simulation::T_MAX)) {
        //----- ADVANCE TIME
        // Don't let simulation time exceed maximum simulation time
        if ((Driver::simTime + Driver::dt) > rp_Simulation::T_MAX) {
            Real   origDt = Driver::dt;
            Driver::dt = (rp_Simulation::T_MAX - Driver::simTime);
            Driver::simTime = rp_Simulation::T_MAX;
            logger.log(  "[Driver] Shortened dt from " + std::to_string(origDt)
                       + " to " + std::to_string(Driver::dt)
                       + " so that tmax=" + std::to_string(rp_Simulation::T_MAX)
                       + " is not exceeded");
        } else {
            Driver::simTime += Driver::dt;
        }
        // TODO: Log as well
        if (rank == MASTER_PE) {
            printf("Step n=%d / t=%.4e / dt=%.4e\n", nStep, Driver::simTime, Driver::dt);
        }

        //----- ADVANCE SOLUTION BASED ON HYDRODYNAMICS
        if (nStep > 1) {
            orchestration::Timer::start("GC Fill");
            grid.fillGuardCells();
            orchestration::Timer::stop("GC Fill");
        }

        //----- ADVANCE SOLUTION
        // Update unk data on interiors only
        double   tStart = MPI_Wtime();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();

            const IntVect       lo        = tileDesc->lo();
            const IntVect       hi        = tileDesc->hi();
            FArray4D            solnData  = tileDesc->data();

            HeatAD::diffusion(solnData,
                              tileDesc->deltas(),
                              1.0,
                              lo,hi);

            HeatAD::solve(solnData,Driver::dt,lo,hi);

        }
        double       wtime_sec = MPI_Wtime() - tStart;

        orchestration::Timer::start("Gather/Write");
        diffusion.logTimestep(nStep, wtime_sec);
        orchestration::Timer::stop("Gather/Write");

        //----- OUTPUT RESULTS TO FILES
        orchestration::Timer::start("Reduce/Write");
        if ((nStep % rp_Driver::WRITE_EVERY_N_STEPS) == 0) {
            grid.writePlotfile(rp_Simulation::NAME + "_plt_" + std::to_string(nStep));
        }
        orchestration::Timer::stop("Reduce/Write");

        //----- UPDATE GRID IF REQUIRED
        // We are running in pseudo-UG for now and can therefore skip this

        //----- COMPUTE dt FOR NEXT STEP
        // NOTE: The AllReduce that follows should appear here
        //       rather than be buried in Driver_computeDt.
        //
        Driver::dt = rp_Driver::DT_AFTER;

        ++nStep;
    }
    orchestration::Timer::stop(rp_Simulation::NAME + " simulation");

    if (Driver::simTime >= rp_Simulation::T_MAX) {
        Logger::instance().log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile(rp_Simulation::NAME + "_plt_final");

    nStep = std::min(nStep, rp_Simulation::MAX_STEPS);

    //----- CLEAN-UP
    // The singletons are finalized automatically when the program is
    // terminating.

    return 0;
}

