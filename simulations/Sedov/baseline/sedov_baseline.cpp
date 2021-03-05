#include <cstdio>
#include <string>
#include <fstream>
#include <iomanip>

#include <mpi.h>

#include "Io.h"
#include "Eos.h"
#include "Hydro.h"
#include "Driver.h"
#include "Simulation.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Timer.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

#include "Flash_par.h"

void createLogfile(const std::string& filename) {
    // Write header to file
    std::ofstream  fptr;
    fptr.open(filename, std::ios::out);
    fptr << "# Testname = Baseline\n";
    fptr << "# NXB = " << NXB << "\n";
    fptr << "# NYB = " << NYB << "\n";
    fptr << "# NZB = " << NZB << "\n";
    fptr << "# N_BLOCKS_X = " << rp_Grid::N_BLOCKS_X << "\n";
    fptr << "# N_BLOCKS_Y = " << rp_Grid::N_BLOCKS_Y << "\n";
    fptr << "# N_BLOCKS_Z = " << rp_Grid::N_BLOCKS_Z << "\n";
    fptr << "# n_distributor_threads = 1 \n";
    fptr << "# n_cpu_threads = 0 \n";
    fptr << "# n_gpu_threads = 0 \n";
    fptr << "# n_blocks_per_packet = 0 \n";
    fptr << "# n_blocks_per_cpu_turn = 0 \n";
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
    if (argc != 2) {
        std::cerr << "One and only one command line argument" << std::endl;
        return 1;
    }
    std::string  filename{argv[1]};

    // TODO: Add in error handling code
    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    orchestration::Logger::instantiate(rp_Simulation::LOG_FILENAME);

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

        createLogfile(filename);
    }

    //----- MIMIC Grid_initDomain
    orchestration::Io&       io      = orchestration::Io::instance();
    orchestration::Grid&     grid    = orchestration::Grid::instance();
    orchestration::Logger&   logger  = orchestration::Logger::instance();

    Driver::dt      = rp_Simulation::DT_INIT;
    Driver::simTime = rp_Simulation::T_0;

    orchestration::Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);
    io.computeLocalIntegralQuantities();
    orchestration::Timer::stop("Set initial conditions");

    //----- OUTPUT RESULTS TO FILES

//    // Compute global integral quantities via DATA MOVEMENT
    orchestration::Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
    // TODO: Shouldn't this be done through the IO unit?
//    grid.writePlotfile(rp_Simulation::NAME + "_plt_ICs");
    orchestration::Timer::stop("Reduce/Write");

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
        
        // For the CSE21 presentation I will time the full time step between
        // data movements.  This should allow me to time the same computation
        // in the GPU runtime version of Sedov.
        // 
        // Each process measures and reports its own walltime for this
        // computation as well as the number of blocks it applied the
        // computation to.
        double   tStart = MPI_Wtime(); 
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();

            const IntVect       lo = tileDesc->lo();
            const IntVect       hi = tileDesc->hi();
            FArray4D            U  = tileDesc->data();

            // FIXME: We should be able to reindex the scratch array so that
            // we aren't needlessly allocating/deallocating scratch blocks.
            IntVect    fHi = IntVect{LIST_NDIM(hi.I()+K1D, hi.J(), hi.K())};
            FArray4D   flX = FArray4D::buildScratchArray4D(lo, fHi, NFLUXES);

            fHi = IntVect{LIST_NDIM(hi.I(), hi.J()+K2D, hi.K())};
            FArray4D   flY = FArray4D::buildScratchArray4D(lo, fHi, NFLUXES);

            fHi = IntVect{LIST_NDIM(hi.I(), hi.J(), hi.K()+K3D)};
            FArray4D   flZ = FArray4D::buildScratchArray4D(lo, fHi, NFLUXES);

            IntVect    cLo = IntVect{LIST_NDIM(lo.I()-K1D, lo.J()-K2D, lo.K()-K3D)};
            IntVect    cHi = IntVect{LIST_NDIM(hi.I()+K1D, hi.J()+K2D, hi.K()+K3D)};
            FArray3D   auxC = FArray3D::buildScratchArray(cLo, cHi);

            hy::computeFluxesHll(Driver::dt, lo, hi,
                                 tileDesc->deltas(),
                                 U, flX, flY, flZ, auxC);
            hy::updateSolutionHll(lo, hi, U, flX, flY, flZ);
            Eos::idealGammaDensIe(lo, hi, U);
        }

        io.computeLocalIntegralQuantities();
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
            logTimestep(filename, nStep, walltimes_sec, blockCounts, nProcs);
        }
        orchestration::Timer::stop("Gather/Write");

        //----- OUTPUT RESULTS TO FILES
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

