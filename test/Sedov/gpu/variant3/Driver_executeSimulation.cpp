#include <cstdio>
#include <string>

#include <mpi.h>

#include <Milhoja_real.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Runtime.h>
#include <Milhoja_Logger.h>
#include <Milhoja_RuntimeBackend.h>

#include "Sedov.h"
#include "RuntimeParameters.h"
#include "Io.h"
#include "Hydro.h"
#include "Timer.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"
#include "DataPacket_gpu_tf_hydro.h"

#include "cpu_tf_ic.h"
#include "Tile_cpu_tf_ic.h"
#include "cpu_tf_IQ.h"
#include "Tile_cpu_tf_IQ.h"
#include "cpu_tf_hydro.h"
#include "Tile_cpu_tf_hydro.h"

/**
 * Numerically approximate using the third GPU variant the solution to the
 * Sedov problem as defined partially by given runtime parameters.  Note that
 * this function initializes and destroys the problem domain.
 */
void    Driver::executeSimulation(void) {
    constexpr int      TIMER_RANK = LEAD_RANK;

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    std::vector<std::string>  variableNames = sim::getVariableNames();

    Io&                      io      = Io::instance();
    RuntimeParameters&       RPs     = RuntimeParameters::instance();
    milhoja::Grid&           grid    = milhoja::Grid::instance();
    milhoja::Logger&         logger  = milhoja::Logger::instance();
    milhoja::Runtime&        runtime = milhoja::Runtime::instance();

    Driver::dt      = RPs.getReal("Simulation", "dtInit");
    Driver::simTime = RPs.getReal("Simulation", "T_0"); 

    //------ ACQUIRE PERSISTENT MILHOJA SCRATCH
    Tile_cpu_tf_IQ::acquireScratch();
    Tile_cpu_tf_hydro::acquireScratch();

    milhoja::RuntimeAction     initBlock_cpu;
    initBlock_cpu.name            = "initBlock_cpu";
    initBlock_cpu.nInitialThreads = RPs.getUnsignedInt("Simulation", "nThreadsForIC");
    initBlock_cpu.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    initBlock_cpu.nTilesPerPacket = 0;
    initBlock_cpu.routine         = cpu_tf_ic::taskFunction;

    // This only makes sense if the iteration is over LEAF blocks.
    milhoja::RuntimeAction     computeIntQuantitiesByBlk;
    computeIntQuantitiesByBlk.name            = "Compute Integral Quantities";
    computeIntQuantitiesByBlk.nInitialThreads = RPs.getUnsignedInt("compIQ_bundle", "nThreadsCpu");
    computeIntQuantitiesByBlk.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    computeIntQuantitiesByBlk.nTilesPerPacket = 0;
    computeIntQuantitiesByBlk.routine         = cpu_tf_IQ::taskFunction;
    const Tile_cpu_tf_IQ    int_IQ_prototype{};

    Timer::start("Set initial conditions");
    Tile_cpu_tf_ic::acquireScratch();
    const Tile_cpu_tf_ic    ic_prototype{};

    grid.initDomain(initBlock_cpu, &ic_prototype);

    Tile_cpu_tf_ic::releaseScratch();
    Timer::stop("Set initial conditions");

    Timer::start("computeLocalIQ");
    runtime.executeCpuTasks("IntegralQ",
                            computeIntQuantitiesByBlk,
                            int_IQ_prototype);
    Timer::stop("computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
//    grid.writePlotfile("sedov_plt_ICs", variableNames);
    Timer::stop("Reduce/Write");

    //----- MIMIC Driver_evolveFlash
    milhoja::RuntimeAction     hydroAdvance_cpu;
    hydroAdvance_cpu.name            = "Advance Hydro Solution - CPU";
    hydroAdvance_cpu.nInitialThreads = RPs.getUnsignedInt("Hydro_cpu/gpu_bundle", "nThreadsCpu");
    hydroAdvance_cpu.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    hydroAdvance_cpu.nTilesPerPacket = 0;
    hydroAdvance_cpu.routine         = cpu_tf_hydro::taskFunction;

    milhoja::RuntimeAction     hydroAdvance_gpu;
    hydroAdvance_gpu.name            = "Advance Hydro Solution - GPU";
    hydroAdvance_gpu.nInitialThreads = RPs.getUnsignedInt("Hydro_cpu/gpu_bundle", "nThreadsGpu");
    hydroAdvance_gpu.teamType        = milhoja::ThreadTeamDataType::SET_OF_BLOCKS;
    hydroAdvance_gpu.nTilesPerPacket = RPs.getUnsignedInt("Hydro_cpu/gpu_bundle", "nBlocksPerPacket");
    hydroAdvance_gpu.routine         = Hydro::advanceSolutionHll_packet_oacc_summit_3;

    // Get RPs that will be used in the loop
    unsigned int      nDistThreads{RPs.getUnsignedInt("Hydro_cpu/gpu_bundle", "nDistributorThreads")};
    milhoja::Real     stagger_usec{RPs.getReal("Hydro_cpu/gpu_bundle", "stagger_usec")};
    unsigned int      nTilesPerCpuTurn{RPs.getUnsignedInt("Hydro_cpu/gpu_bundle", "nTilesPerCpuTurn")};

    ProcessTimer  hydro{"sedov_timings.dat", "GPU",
                        nDistThreads,
                        stagger_usec,
                        hydroAdvance_cpu.nInitialThreads,
                        hydroAdvance_gpu.nInitialThreads,
                        hydroAdvance_gpu.nTilesPerPacket,
                        nTilesPerCpuTurn,
                        GLOBAL_COMM, TIMER_RANK};

    Timer::start("sedov simulation");

    unsigned int      nStep{1};
    unsigned int      maxSteps{RPs.getUnsignedInt("Simulation", "maxSteps")};
    milhoja::Real     tMax{RPs.getReal("Simulation", "tMax")};
    milhoja::Real     dtAfter{RPs.getReal("Driver", "dtAfter")};
    unsigned int      writeEveryNSteps{RPs.getUnsignedInt("Driver", "writeEveryNSteps")};
    while ((nStep <= maxSteps) && (Driver::simTime < tMax)) {
        //----- ADVANCE TIME
        // Don't let simulation time exceed maximum simulation time
        if ((Driver::simTime + Driver::dt) > tMax) {
            milhoja::Real   origDt = Driver::dt;
            Driver::dt = (tMax - Driver::simTime);
            Driver::simTime = tMax;
            logger.log(  "[Driver] Shortened dt from " + std::to_string(origDt)
                       + " to " + std::to_string(Driver::dt)
                       + " so that tmax=" + std::to_string(tMax)
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
        const Tile_cpu_tf_hydro          cpu_tf_hydro_prototype{Driver::dt};
	    const DataPacket_gpu_tf_hydro    gpu_tf00_prototype{Driver::dt};
        runtime.executeCpuGpuSplitTasks("Advance Hydro Solution",
                                        nDistThreads,
                                        stagger_usec,
                                        hydroAdvance_cpu,
                                        cpu_tf_hydro_prototype,
                                        hydroAdvance_gpu,
                                        gpu_tf00_prototype,
                                        nTilesPerCpuTurn);
//        runtime.executeCpuGpuSplitTasks_timed("Advance Hydro Solution",
//                                              nDistThreads,
//                                              stagger_usec,
//                                              hydroAdvance_cpu,
//                                              cpu_tf00_prototype,
//                                              hydroAdvance_gpu,
//                                              gpu_tf00_prototype,
//                                              nTilesPerCpuTurn,
//                                              nStep,
//                                              GLOBAL_COMM);
// TODO: Currently, the test is only using the GPU because it makes testing easier,
//       and there's no need to account for roundoff error. Eventually, we should
//       move back to using the CpuGpuSplit thread team config.
//        runtime.executeGpuTasks("Advance Hydro Solution",
//                                nDistThreads,
//                                stagger_usec,
//                                hydroAdvance_gpu,
//                                gpu_tf00_prototype);

        double   wtime_sec = MPI_Wtime() - tStart;
        Timer::start("Gather/Write");
        hydro.logTimestep(nStep, wtime_sec);
        Timer::stop("Gather/Write");

        Timer::start("computeLocalIQ");
        runtime.executeCpuTasks("IntegralQ",
                                computeIntQuantitiesByBlk,
                                int_IQ_prototype);
        Timer::stop("computeLocalIQ");

        //----- OUTPUT RESULTS TO FILES
        //  local integral quantities computed as part of previous bundle
        Timer::start("Reduce/Write");
        io.reduceToGlobalIntegralQuantities();
        io.writeIntegralQuantities(Driver::simTime);

        if ((nStep % writeEveryNSteps) == 0) {
            grid.writePlotfile("sedov_plt_" + std::to_string(nStep),
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
        Driver::dt = dtAfter;

        // FIXME: This is a cheap hack necessitated by the fact that the runtime
        // does not yet have a real memory manager.
        milhoja::RuntimeBackend::instance().reset();

        ++nStep;
    }
    Timer::stop("sedov simulation");

    //----- RELEASE PERSISTENT MILHOJA Scratch Memory
    // TODO: It would be good to include this is performance timings since it
    // is extra overhead associated with Milhoja use
    Tile_cpu_tf_hydro::releaseScratch();
    Tile_cpu_tf_IQ::releaseScratch();

    if (Driver::simTime >= tMax) {
        logger.log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile("sedov_plt_final", variableNames);

    nStep = std::min(nStep, maxSteps);

    grid.destroyDomain();
}

