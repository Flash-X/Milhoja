#include <cstdio>
#include <string>

#include <mpi.h>

#include "Io.h"
#include "Hydro.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"
#include "DataPacket_Hydro_gpu_1.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Timer.h"
#include "Runtime.h"
#include "Backend.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

#include "Flash_par.h"

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

    // Analogous to calling IO_init
    orchestration::Io::instantiate(rp_Simulation::INTEGRAL_QUANTITIES_FILENAME);

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

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
    computeIntQuantitiesByBlk.nInitialThreads = rp_Bundle_1::N_THREADS_CPU;
    computeIntQuantitiesByBlk.teamType        = ThreadTeamDataType::BLOCK;
    computeIntQuantitiesByBlk.nTilesPerPacket = 0;
    computeIntQuantitiesByBlk.routine         
        = ActionRoutines::Io_computeIntegralQuantitiesByBlock_tile_cpu;

    orchestration::Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);
    orchestration::Timer::stop("Set initial conditions");

    orchestration::Timer::start("computeLocalIQ");
    runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
    orchestration::Timer::stop("computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    orchestration::Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
//    grid.writePlotfile(rp_Simulation::NAME + "_plt_ICs");
    orchestration::Timer::stop("Reduce/Write");

    //----- MIMIC Driver_evolveFlash
    RuntimeAction     hydroAdvance_cpu;
    hydroAdvance_cpu.name            = "Advance Hydro Solution - CPU";
    hydroAdvance_cpu.nInitialThreads = rp_Bundle_2::N_THREADS_CPU;
    hydroAdvance_cpu.teamType        = ThreadTeamDataType::BLOCK;
    hydroAdvance_cpu.nTilesPerPacket = 0;
    hydroAdvance_cpu.routine         = Hydro::advanceSolutionHll_tile_cpu;

    RuntimeAction     hydroAdvance_gpu;
    hydroAdvance_gpu.name            = "Advance Hydro Solution - GPU";
    hydroAdvance_gpu.nInitialThreads = rp_Bundle_2::N_THREADS_GPU;
    hydroAdvance_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    hydroAdvance_gpu.nTilesPerPacket = rp_Bundle_2::N_BLOCKS_PER_PACKET;
    hydroAdvance_gpu.routine         = Hydro::advanceSolutionHll_packet_oacc_summit_1;

    ProcessTimer  hydro{rp_Simulation::NAME + "_timings.dat", "GPU",
                        rp_Bundle_2::N_DISTRIBUTOR_THREADS,
                        rp_Bundle_2::STAGGER_USEC,
                        hydroAdvance_cpu.nInitialThreads,
                        hydroAdvance_gpu.nInitialThreads,
                        hydroAdvance_gpu.nTilesPerPacket,
                        rp_Bundle_2::N_TILES_PER_CPU_TURN};

    orchestration::Timer::start(rp_Simulation::NAME + " simulation");

    unsigned int   nStep   = 1;

    const DataPacket_Hydro_gpu_1    packetPrototype;
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
//        runtime.executeCpuGpuSplitTasks("Advance Hydro Solution",
//                                        rp_Bundle_2::N_DISTRIBUTOR_THREADS,
//                                        hydroAdvance_cpu,
//                                        hydroAdvance_gpu,
//                                        packetPrototype,
//                                        rp_Bundle_2::STAGGER_USEC,
//                                        rp_Bundle_2::N_TILES_PER_CPU_TURN);
        runtime.executeCpuGpuSplitTasks_timed("Advance Hydro Solution",
                                              rp_Bundle_2::N_DISTRIBUTOR_THREADS,
                                              hydroAdvance_cpu,
                                              hydroAdvance_gpu,
                                              packetPrototype,
                                              rp_Bundle_2::STAGGER_USEC,
                                              rp_Bundle_2::N_TILES_PER_CPU_TURN,
                                              nStep);
        double   wtime_sec = MPI_Wtime() - tStart;
        orchestration::Timer::start("Gather/Write");
        hydro.logTimestep(nStep, wtime_sec);
        orchestration::Timer::stop("Gather/Write");

        orchestration::Timer::start("computeLocalIQ");
        runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
        orchestration::Timer::stop("computeLocalIQ");

        //----- OUTPUT RESULTS TO FILES
        //  local integral quantities computed as part of previous bundle
        orchestration::Timer::start("Reduce/Write");
        io.reduceToGlobalIntegralQuantities();
        io.writeIntegralQuantities(Driver::simTime);

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

        // FIXME: This is a cheap hack necessitated by the fact that the runtime
        // does not yet have a real memory manager.
        orchestration::Backend::instance().reset();

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

