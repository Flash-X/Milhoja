#include <cstdio>
#include <string>

#include <mpi.h>

#include <Milhoja_real.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#include <Milhoja_Logger.h>

#include "Sedov.h"
#include "Io.h"
#include "Hydro.h"
#include "Timer.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"
#include "loadGridConfiguration.h"
#include "DataPacket_Hydro_gpu_2.h"
#include "errorEstBlank.h"

#include "Flash_par.h"

constexpr int   LOG_RANK   = LEAD_RANK;
constexpr int   IO_RANK    = LEAD_RANK;
constexpr int   TIMER_RANK = LEAD_RANK;

int main(int argc, char* argv[]) {
    // TODO: Add in error handling code

    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    milhoja::Logger::instantiate(rp_Simulation::LOG_FILENAME,
                                 GLOBAL_COMM, LOG_RANK);

    // Analogous to calling Orchestration_init
    milhoja::Runtime::instantiate(rp_Runtime::N_THREAD_TEAMS, 
                                  rp_Runtime::N_THREADS_PER_TEAM,
                                  rp_Runtime::N_STREAMS,
                                  rp_Runtime::MEMORY_POOL_SIZE_BYTES);

    // Analogous to calling Grid_init
    loadGridConfiguration();
    milhoja::Grid::instantiate();

    // Analogous to calling IO_init
    Io::instantiate(rp_Simulation::INTEGRAL_QUANTITIES_FILENAME,
                    GLOBAL_COMM, IO_RANK);

    // Analogous to calling sim_init
    std::vector<std::string>  variableNames = sim::getVariableNames();

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    //----- MIMIC Grid_initDomain
    Io&                      io      = Io::instance();
    milhoja::Grid&           grid    = milhoja::Grid::instance();
    milhoja::Logger&         logger  = milhoja::Logger::instance();
    milhoja::Runtime&        runtime = milhoja::Runtime::instance();

    Driver::dt      = rp_Simulation::DT_INIT;
    Driver::simTime = rp_Simulation::T_0;

    // This only makes sense if the iteration is over LEAF blocks.
    milhoja::RuntimeAction     computeIntQuantitiesByBlk;
    computeIntQuantitiesByBlk.name            = "Compute Integral Quantities";
    computeIntQuantitiesByBlk.nInitialThreads = rp_Bundle_1::N_THREADS_CPU;
    computeIntQuantitiesByBlk.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    computeIntQuantitiesByBlk.nTilesPerPacket = 0;
    computeIntQuantitiesByBlk.routine         
        = ActionRoutines::Io_computeIntegralQuantitiesByBlock_tile_cpu;

    Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);
    Timer::stop("Set initial conditions");

    Timer::start("computeLocalIQ");
    runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
    Timer::stop("computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
//    grid.writePlotfile(rp_Simulation::NAME + "_plt_ICs", variableNames);
    Timer::stop("Reduce/Write");

    //----- MIMIC Driver_evolveFlash
    milhoja::RuntimeAction     hydroAdvance_cpu;
    hydroAdvance_cpu.name            = "Advance Hydro Solution - CPU";
    hydroAdvance_cpu.nInitialThreads = rp_Bundle_2::N_THREADS_CPU;
    hydroAdvance_cpu.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    hydroAdvance_cpu.nTilesPerPacket = 0;
    hydroAdvance_cpu.routine         = Hydro::advanceSolutionHll_tile_cpu;

    milhoja::RuntimeAction     hydroAdvance_gpu;
    hydroAdvance_gpu.name            = "Advance Hydro Solution - GPU";
    hydroAdvance_gpu.nInitialThreads = rp_Bundle_2::N_THREADS_GPU;
    hydroAdvance_gpu.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
    hydroAdvance_gpu.nTilesPerPacket = rp_Bundle_2::N_BLOCKS_PER_PACKET;
    hydroAdvance_gpu.routine         = Hydro::advanceSolutionHll_packet_oacc_summit_2;

    ProcessTimer  hydro{rp_Simulation::NAME + "_timings.dat", "GPU",
                        rp_Bundle_2::N_DISTRIBUTOR_THREADS,
                        rp_Bundle_2::STAGGER_USEC,
                        hydroAdvance_cpu.nInitialThreads,
                        hydroAdvance_gpu.nInitialThreads,
                        hydroAdvance_gpu.nTilesPerPacket,
                        rp_Bundle_2::N_TILES_PER_CPU_TURN,
                        GLOBAL_COMM, TIMER_RANK};

    Timer::start(rp_Simulation::NAME + " simulation");

    unsigned int   nStep   = 1;

    const DataPacket_Hydro_gpu_2    packetPrototype;
    while ((nStep <= rp_Simulation::MAX_STEPS) && (Driver::simTime < rp_Simulation::T_MAX)) {
        //----- ADVANCE TIME
        // Don't let simulation time exceed maximum simulation time
        if ((Driver::simTime + Driver::dt) > rp_Simulation::T_MAX) {
            milhoja::Real   origDt = Driver::dt;
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
        if (rank == LEAD_RANK) {
            printf("Step n=%d / t=%.4e / dt=%.4e\n", nStep, Driver::simTime, Driver::dt);
        }

        //----- ADVANCE SOLUTION BASED ON HYDRODYNAMICS
        if (nStep > 1) {
            Timer::start("GC Fill");
            grid.fillGuardCells();
            Timer::stop("GC Fill");
        }

        double   tStart = MPI_Wtime();
//        runtime.executeCpuGpuSplitTasks("Advance Hydro Solution",
//                                        rp_Bundle_2::N_DISTRIBUTOR_THREADS,
//                                        rp_Bundle_2::STAGGER_USEC,
//                                        hydroAdvance_cpu,
//                                        hydroAdvance_gpu,
//                                        packetPrototype,
//                                        rp_Bundle_2::N_TILES_PER_CPU_TURN);
        runtime.executeCpuGpuSplitTasks_timed("Advance Hydro Solution",
                                              rp_Bundle_2::N_DISTRIBUTOR_THREADS,
                                              rp_Bundle_2::STAGGER_USEC,
                                              hydroAdvance_cpu,
                                              hydroAdvance_gpu,
                                              packetPrototype,
                                              rp_Bundle_2::N_TILES_PER_CPU_TURN,
                                              nStep,
                                              GLOBAL_COMM);
        double   wtime_sec = MPI_Wtime() - tStart;
        Timer::start("Gather/Write");
        hydro.logTimestep(nStep, wtime_sec);
        Timer::stop("Gather/Write");

        Timer::start("computeLocalIQ");
        runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
        Timer::stop("computeLocalIQ");

        //----- OUTPUT RESULTS TO FILES
        //  local integral quantities computed as part of previous bundle
        Timer::start("Reduce/Write");
        io.reduceToGlobalIntegralQuantities();
        io.writeIntegralQuantities(Driver::simTime);

        if ((nStep % rp_Driver::WRITE_EVERY_N_STEPS) == 0) {
            grid.writePlotfile(rp_Simulation::NAME + "_plt_" + std::to_string(nStep),
                               variableNames);
        }
        Timer::stop("Reduce/Write");

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
        milhoja::RuntimeBackend::instance().reset();

        ++nStep;
    }
    Timer::stop(rp_Simulation::NAME + " simulation");

    if (Driver::simTime >= rp_Simulation::T_MAX) {
        logger.log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile(rp_Simulation::NAME + "_plt_final", variableNames);

    nStep = std::min(nStep, rp_Simulation::MAX_STEPS);

    //----- CLEAN-UP
    // The singletons are finalized automatically when the program is
    // terminating.

    return 0;
}

