#include <cstdio>
#include <string>
#include <fstream>
#include <iomanip>

#include <mpi.h>

#include "Io.h"
#include "Hydro.h"
#include "Driver.h"
#include "Simulation.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Timer.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

#include "Flash_par.h"

void createLogfile(const std::string& filename,
                   const unsigned int nCpuThreads) {
    // Write header to file
    std::ofstream  fptr;
    fptr.open(filename, std::ios::out);
    fptr << "# Testname = CPU\n";
    fptr << "# NXB = " << NXB << "\n";
    fptr << "# NYB = " << NYB << "\n";
    fptr << "# NZB = " << NZB << "\n";
    fptr << "# N_BLOCKS_X = " << rp_Grid::N_BLOCKS_X << "\n";
    fptr << "# N_BLOCKS_Y = " << rp_Grid::N_BLOCKS_Y << "\n";
    fptr << "# N_BLOCKS_Z = " << rp_Grid::N_BLOCKS_Z << "\n";
    fptr << "# n_distributor_threads = 1\n";
    fptr << "# n_cpu_threads = " << nCpuThreads << "\n";
    fptr << "# n_gpu_threads = 0\n";
    fptr << "# n_blocks_per_packet = 0\n";
    fptr << "# n_blocks_per_cpu_turn = 0\n";
    fptr << "# MPI_Wtick_sec = " << MPI_Wtick() << "\n";
    fptr << "# step,nblocks_1,walltime_sec_1,...,nblocks_N,walltime_sec_N\n";
    fptr.close();
}

void logTimestep(const std::string& filename,
                 const unsigned int step,
                 const double* walltimes_sec,
                 const unsigned int* blockCounts,
                 const int nProcs) {
    std::ofstream  fptr;
    fptr.open(filename, std::ios::out | std::ios::app);
    fptr << std::setprecision(15) 
         << step << ",";
    for (int rank=0; rank<nProcs; ++rank) {
        fptr << blockCounts[rank] << ',' << walltimes_sec[rank];
        if (rank < nProcs - 1) {
            fptr << ',';
        }
    }
    fptr << std::endl;
    fptr.close();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Two and only two command line arguments\n";
        return 1;
    }

    std::string  fname_hydro{argv[1]};
    std::string  fname_integrate{argv[2]};
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
    int  nProcs = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);
    MPI_Comm_size(GLOBAL_COMM, &nProcs);

    double*       walltimes_sec = nullptr;
    unsigned int* blockCounts = nullptr;
    if (rank == MASTER_PE) {
        walltimes_sec = new double[nProcs];
        blockCounts   = new unsigned int[nProcs];
    }

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

    orchestration::Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);
    // Compute local integral quantities
    runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
    orchestration::Timer::stop("Set initial conditions");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    orchestration::Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
    // TODO: Shouldn't this be done through the IO unit?
    // FIXME: We disable this for testing as the plot files might be very large
//    grid.writePlotfile(rp_Simulation::NAME + "_plt_ICs");
    orchestration::Timer::stop("Reduce/Write");

    //----- MIMIC Driver_evolveFlash
    RuntimeAction     hydroAdvance;
    hydroAdvance.name            = "Advance Hydro Solution";
    hydroAdvance.nInitialThreads = rp_Hydro::N_THREADS_FOR_ADV_SOLN;
    hydroAdvance.teamType        = ThreadTeamDataType::BLOCK;
    hydroAdvance.nTilesPerPacket = 0;
    hydroAdvance.routine         = Hydro::advanceSolutionHll_tile_cpu;

    if (rank == MASTER_PE) {
        createLogfile(fname_hydro, hydroAdvance.nInitialThreads);
        createLogfile(fname_integrate, 
                      computeIntQuantitiesByBlk.nInitialThreads);
    }

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

        // For the CSE21 presentation, time each action separetely to
        // gather information about walltime/block as a function of 
        // number of threads.  This should help tune the runtime
        // configuration.
        // 
        // Each process measures and reports its own walltime for this
        // computation as well as the number of blocks it applied the
        // computation to.
        double   tStart = MPI_Wtime(); 
        runtime.executeCpuTasks("Advance Hydro Solution", hydroAdvance);
        double       wtime_sec = MPI_Wtime() - tStart;
        orchestration::Timer::start("Gather/Write");
        unsigned int nBlocks   = grid.getNumberLocalBlocks();
        MPI_Gather(&wtime_sec, 1, MPI_DOUBLE,
                   walltimes_sec, 1, MPI_DOUBLE, MASTER_PE,
                   GLOBAL_COMM);
        MPI_Gather(&nBlocks, 1, MPI_UNSIGNED,
                   blockCounts, 1, MPI_UNSIGNED, MASTER_PE,
                   GLOBAL_COMM);
        if (rank == MASTER_PE) {
            logTimestep(fname_hydro, nStep, walltimes_sec, blockCounts, nProcs);
        }
        orchestration::Timer::stop("Gather/Write");

        //----- OUTPUT RESULTS TO FILES
        // Compute local integral quantities
        // TODO: This should be run as a CPU-based pipeline extension
        //       to the physics action bundle.
        tStart = MPI_Wtime(); 
        runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
        wtime_sec = MPI_Wtime() - tStart;
        orchestration::Timer::start("Gather/Write");
        MPI_Gather(&wtime_sec, 1, MPI_DOUBLE,
                   walltimes_sec, 1, MPI_DOUBLE, MASTER_PE,
                   GLOBAL_COMM);
        if (rank == MASTER_PE) {
            logTimestep(fname_integrate, nStep, walltimes_sec, blockCounts, nProcs);
        }
        orchestration::Timer::stop("Gather/Write");

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
    if (rank == MASTER_PE) {
        delete [] walltimes_sec;
        walltimes_sec = nullptr;
        delete [] blockCounts;
        blockCounts = nullptr;
    }

    return 0;
}

