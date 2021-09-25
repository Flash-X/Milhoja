#include <cstdio>
#include <string>

#include <mpi.h>

#include "Io.h"
#include "Hydro.h"
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

    //----- APPLICATIONS MANAGE MPI
    // The idea is to follow xSDK M3
    MPI_Init(&argc, &argv);
    MPI_Comm     MILHOJA_MPI_COMM = MPI_COMM_WORLD;

    // Test that Milhoja uses the communicator correctly
//    int globalRank = -1;
//    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
//
//    // This results in the first ranks having no communicator
////    int color = MPI_UNDEFINED;
//    // Split into two communicators, each of which has its own 
//    // dedicated AMReX setup
//    int color = 0;
//    if (globalRank >= 2) {
//        color = 1;
//    }
//    MPI_Comm_split(MPI_COMM_WORLD, color, globalRank, &MILHOJA_MPI_COMM);

    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    // TODO: If we use more than one communicator, each communicator should
    // probably have its own log file.
    orchestration::Logger::instantiate(MILHOJA_MPI_COMM,
                                       rp_Simulation::LOG_FILENAME);

    // Analogous to calling Orchestration_init
    // The runtime doesn't need the communicator.  Rather, this application
    // should setup and call the runtime based on the communicator.
    orchestration::Runtime::instantiate(rp_Runtime::N_THREAD_TEAMS, 
                                        rp_Runtime::N_THREADS_PER_TEAM,
                                        rp_Runtime::N_STREAMS,
                                        rp_Runtime::MEMORY_POOL_SIZE_BYTES);

    // Analogous to calling Grid_init
    // Each communicator gets its own dedicated, independent copy of Grid (TBC)
    orchestration::Grid::instantiate(MILHOJA_MPI_COMM,
                                     rp_Grid::X_MIN, rp_Grid::X_MAX,
                                     rp_Grid::Y_MIN, rp_Grid::Y_MAX,
                                     rp_Grid::Z_MIN, rp_Grid::Z_MAX,
                                     NXB, NYB, NZB,
                                     rp_Grid::N_BLOCKS_X,
                                     rp_Grid::N_BLOCKS_Y,
                                     rp_Grid::N_BLOCKS_Z,
                                     rp_Grid::LREFINE_MAX,
                                     NGUARD, NUNKVAR,
                                     Simulation::setInitialConditions_tile_cpu,
                                     Simulation::errorEstBlank);

    // Analogous to calling IO_init
    // Since each communicator has its own copy of Grid, we want to compute the
    // global IQ for each communicator.  Therefore, each communicator should
    // have its own data file.
    orchestration::Io::instantiate(MILHOJA_MPI_COMM,
                                   rp_Simulation::INTEGRAL_QUANTITIES_FILENAME);

    int  rank = 0;
    MPI_Comm_rank(MILHOJA_MPI_COMM, &rank);

    //----- MIMIC Grid_initDomain
    orchestration::Io&       io      = orchestration::Io::instance();
    orchestration::Grid&     grid    = orchestration::Grid::instance();
    orchestration::Logger&   logger  = orchestration::Logger::instance();
    orchestration::Runtime&  runtime = orchestration::Runtime::instance();

    Driver::dt      = rp_Simulation::DT_INIT;
    Driver::simTime = rp_Simulation::T_0;

    // This only makes sense if the iteration is over LEAF blocks.
    RuntimeAction     computeIntQuantitiesByBlk;
    computeIntQuantitiesByBlk.name            = "Compute Integral Quantities";
    computeIntQuantitiesByBlk.nInitialThreads = rp_Io::N_THREADS_FOR_INT_QUANTITIES;
    computeIntQuantitiesByBlk.teamType        = ThreadTeamDataType::BLOCK;
    computeIntQuantitiesByBlk.nTilesPerPacket = 0;
    computeIntQuantitiesByBlk.routine         
        = ActionRoutines::Io_computeIntegralQuantitiesByBlock_tile_cpu;

    // The Timer calls put in a barrier and then log the walltime.  Therefore,
    // this allows us to time each communicator's use of the runtime
    // independently.  If each communicator had a different filename, then this
    // should work nicely.
    orchestration::Timer::start(MILHOJA_MPI_COMM, "Set initial conditions");
    grid.initDomain(rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC);
    orchestration::Timer::stop(MILHOJA_MPI_COMM, "Set initial conditions");

    orchestration::Timer::start(MILHOJA_MPI_COMM, "computeLocalIQ");
    runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
    orchestration::Timer::stop(MILHOJA_MPI_COMM, "computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    orchestration::Timer::start(MILHOJA_MPI_COMM, "Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
//    grid.writePlotfile(rp_Simulation::NAME + "_plt_ICs");
    orchestration::Timer::stop(MILHOJA_MPI_COMM, "Reduce/Write");

    //----- MIMIC Driver_evolveFlash
    RuntimeAction     hydroAdvance;
    hydroAdvance.name            = "Advance Hydro Solution";
    hydroAdvance.nInitialThreads = rp_Hydro::N_THREADS_FOR_ADV_SOLN;
    hydroAdvance.teamType        = ThreadTeamDataType::BLOCK;
    hydroAdvance.nTilesPerPacket = 0;
    hydroAdvance.routine         = Hydro::advanceSolutionHll_tile_cpu;

    // The ProcessTimer gathers data from all MPI processes and then logs each
    // result.  Therefore, this timing is associated to a particular use of the
    // runtime and it makes sense to time each communicator independently.  Each
    // communicator should have a different data file if using more than one
    // communicator.
    ProcessTimer  hydro{MILHOJA_MPI_COMM,
                        rp_Simulation::NAME + "_timings.dat", "CPU",
                        N_DIST_THREADS, 0,
                        hydroAdvance.nInitialThreads,
                        N_GPU_THREADS,
                        N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN};

    orchestration::Timer::start(MILHOJA_MPI_COMM, rp_Simulation::NAME + " simulation");

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
            orchestration::Timer::start(MILHOJA_MPI_COMM, "GC Fill");
            grid.fillGuardCells();
            orchestration::Timer::stop(MILHOJA_MPI_COMM, "GC Fill");
        }

        double   tStart = MPI_Wtime();
        runtime.executeCpuTasks("Advance Hydro Solution", hydroAdvance);
        double   wtime_sec = MPI_Wtime() - tStart;
        orchestration::Timer::start(MILHOJA_MPI_COMM, "Gather/Write");
        hydro.logTimestep(nStep, wtime_sec);
        orchestration::Timer::stop(MILHOJA_MPI_COMM, "Gather/Write");

        orchestration::Timer::start(MILHOJA_MPI_COMM, "computeLocalIQ");
        runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
        orchestration::Timer::stop(MILHOJA_MPI_COMM, "computeLocalIQ");

        //----- OUTPUT RESULTS TO FILES
        // Compute local integral quantities
        // TODO: This should be run as a CPU-based pipeline extension
        //       to the physics action bundle.
        orchestration::Timer::start(MILHOJA_MPI_COMM, "Reduce/Write");
        io.reduceToGlobalIntegralQuantities();
        io.writeIntegralQuantities(Driver::simTime);

        if ((nStep % rp_Driver::WRITE_EVERY_N_STEPS) == 0) {
            grid.writePlotfile(rp_Simulation::NAME + "_plt_" + std::to_string(nStep));
        }
        orchestration::Timer::stop(MILHOJA_MPI_COMM, "Reduce/Write");

        //----- UPDATE GRID IF REQUIRED
        // We are running in pseudo-UG for now and can therefore skip this

        //----- COMPUTE dt FOR NEXT STEP
        // NOTE: The AllReduce that follows should appear here
        //       rather than be buried in Driver_computeDt.
        //
        // When this problem is run in FLASH-X, the hydro dt is always greater
        // than 5.0e-5 seconds.  Therefore, using a dt value fixed to a smaller
        // value should always keep us on the stable side of the CFL condition.
        // Therefore, we skip the computeDt for Hydro here. 
        //
        // When a dt value of 5.0e-5 is used, FLASH-X complains that it is too
        // low and sets dt to the Hydro CFL-determined dt value, which should be 
        // Simulation::DT_INIT.  There after, it allows for 5.0e-5.  Therefore,
        // we mimic that dt sequence here so that we can directly compare
        // results.
        Driver::dt = rp_Driver::DT_AFTER;

        ++nStep;
    }
    orchestration::Timer::stop(MILHOJA_MPI_COMM, rp_Simulation::NAME + " simulation");

    if (Driver::simTime >= rp_Simulation::T_MAX) {
        Logger::instance().log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile(rp_Simulation::NAME + "_plt_final");

    nStep = std::min(nStep, rp_Simulation::MAX_STEPS);

    //----- CLEAN-UP
    // The singletons are finalized automatically when the program is
    // terminating.
    grid.finalize();

//    MPI_Comm_free(&MILHOJA_MPI_COMM);
    MPI_Finalize();

    return 0;
}

