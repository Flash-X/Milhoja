#include <cstdio>
#include <string>

#include <mpi.h>

#include "HeatADAction.h"
#include "HeatAD.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Timer.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

#include "Flash_par.h"

constexpr  unsigned int N_DIST_THREADS      = 1;
constexpr  unsigned int N_GPU_THREADS       = 0;
constexpr  unsigned int N_BLKS_PER_PACKET   = 0;
constexpr  unsigned int N_BLKS_PER_CPU_TURN = 1;

int main(int argc, char* argv[]) {
    // TODO: Add in error handling code

    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    orchestration::Logger::instantiate(rp_Simulation::LOG_FILENAME);

    // Analogous to calling Orchestration_init
    orchestration::Runtime::instantiate(rp_Runtime::N_THREAD_TEAMS, 
                                        rp_Runtime::N_THREADS_PER_TEAM,
                                        rp_Runtime::N_STREAMS,
                                        rp_Runtime::MEMORY_POOL_SIZE_BYTES);

    // Analogous to calling Grid_init
    orchestration::Grid::instantiate();

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    //----- MIMIC Grid_initDomain
    orchestration::Grid&     grid    = orchestration::Grid::instance();
    orchestration::Logger&   logger  = orchestration::Logger::instance();
    orchestration::Runtime&  runtime = orchestration::Runtime::instance();

    Driver::dt      = rp_Simulation::DT_INIT;
    Driver::simTime = rp_Simulation::T_0;
    HeatAD::alpha   = rp_HeatAD::ALPHA;

    orchestration::Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);
    orchestration::Timer::stop("Set initial conditions");

    //----- MIMIC Driver_evolveFlash
    RuntimeAction     heatAdvance;
    heatAdvance.name            = "Advance HeatAD Solution";
    heatAdvance.nInitialThreads = rp_HeatADAction::N_THREADS_FOR_ADV_SOLN;
    heatAdvance.teamType        = ThreadTeamDataType::BLOCK;
    heatAdvance.nTilesPerPacket = 0;
    heatAdvance.routine         = HeatADAction::advanceSolution_tile_cpu;

    ProcessTimer  heatdiffusion{rp_Simulation::NAME + "_timings_heatdiffusion.dat", "CPU",
                                N_DIST_THREADS,
                                heatAdvance.nInitialThreads,
                                N_GPU_THREADS,
                                N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN};

    orchestration::Timer::start(rp_Simulation::NAME + " simulation");

    unsigned int   nStep   = 1;
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

        double   tStart = MPI_Wtime();
        runtime.executeCpuTasks("Advance HeatAD Solution", heatAdvance);
        double   wtime_sec = MPI_Wtime() - tStart;
        orchestration::Timer::start("Gather/Write");
        heatdiffusion.logTimestep(nStep, wtime_sec);
        orchestration::Timer::stop("Gather/Write");

        //----- OUTPUT RESULTS TO FILES
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

