#include <cstdio>
#include <string>

#include <mpi.h>

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Runtime.h>
#include <Milhoja_Logger.h>

#include "Sedov.h"
#include "RuntimeParameters.h"
#include "Io.h"
#include "Hydro.h"
#include "Timer.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"

// TODO: This would need to be inserted by a Driver code generator
#if MILHOJA_NDIM == 2
#include "cpu_tf00_2D.h"
#else
#include "cpu_tf00_3D.h"
#include "Tile_cpu_tf00_3D.h"
#endif

void    Driver::executeSimulation(void) {
    constexpr  int          TIMER_RANK          = LEAD_RANK;
    constexpr  unsigned int N_DIST_THREADS      = 1;
    constexpr  unsigned int N_GPU_THREADS       = 0;
    constexpr  unsigned int N_BLKS_PER_PACKET   = 0;
    constexpr  unsigned int N_BLKS_PER_CPU_TURN = 1;

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

    milhoja::RuntimeAction     initBlock_cpu;
    initBlock_cpu.name            = "initBlock_cpu";
    initBlock_cpu.nInitialThreads = RPs.getUnsignedInt("Simulation", "nThreadsForIC");
    initBlock_cpu.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    initBlock_cpu.nTilesPerPacket = 0;
    initBlock_cpu.routine         = Simulation::setInitialConditions_tile_cpu;

    // This only makes sense if the iteration is over LEAF blocks.
    milhoja::RuntimeAction     computeIntQuantitiesByBlk;
    computeIntQuantitiesByBlk.name            = "Compute Integral Quantities";
    computeIntQuantitiesByBlk.nInitialThreads = RPs.getUnsignedInt("Io", "nThreadsForIntQuantities");
    computeIntQuantitiesByBlk.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    computeIntQuantitiesByBlk.nTilesPerPacket = 0;
    computeIntQuantitiesByBlk.routine         
        = ActionRoutines::Io_computeIntegralQuantitiesByBlock_tile_cpu;
    const milhoja::TileWrapper int_IQ_prototype{};

    Timer::start("Set initial conditions");
    grid.initDomain(initBlock_cpu);
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
    milhoja::RuntimeAction     hydroAdvance;
    hydroAdvance.name            = "Advance Hydro Solution";
    hydroAdvance.nInitialThreads = RPs.getUnsignedInt("Hydro", "nThreadsForAdvanceSolution");
    hydroAdvance.teamType        = milhoja::ThreadTeamDataType::BLOCK;
    hydroAdvance.nTilesPerPacket = 0;
    // TODO: This would need to be inserted by a Driver code generator
#if MILHOJA_NDIM == 2
    hydroAdvance.routine         = cpu_tf00_2D::taskFunction;
#else
    hydroAdvance.routine         = cpu_tf00_3D::taskFunction;
#endif

    ProcessTimer  hydro{"sedov_timings.dat", "CPU",
                        N_DIST_THREADS, 0,
                        hydroAdvance.nInitialThreads,
                        N_GPU_THREADS,
                        N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN,
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
        const Tile_cpu_tf00_3D   cpu_tf00_prototype{Driver::dt};
        runtime.executeCpuTasks("Advance Hydro Solution",
                                hydroAdvance, cpu_tf00_prototype);
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
        // Compute local integral quantities
        // TODO: This should be run as a CPU-based pipeline extension
        //       to the physics action bundle.
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

        ++nStep;
    }
    Timer::stop("sedov simulation");

    if (Driver::simTime >= tMax) {
        logger.log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile("sedov_plt_final", variableNames);

    nStep = std::min(nStep, maxSteps);

    grid.destroyDomain();
}

