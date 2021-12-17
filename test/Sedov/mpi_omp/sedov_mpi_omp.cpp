#include <cstdio>
#include <string>

#include <mpi.h>

#include "milhoja.h"
#include "Grid_REAL.h"
#include "Grid.h"
#include "Timer.h"
#include "OrchestrationLogger.h"

#include "Io.h"
#include "Eos.h"
#include "Hydro.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"
#include "loadGridConfiguration.h"

#include "errorEstBlank.h"

#include "Sedov.h"
#include "Flash_par.h"

constexpr  int          LOG_RANK            = LEAD_RANK;
constexpr  int          IO_RANK             = LEAD_RANK;
constexpr  int          TIMER_RANK          = LEAD_RANK;

constexpr  unsigned int N_DIST_THREADS      = 0;
constexpr  unsigned int N_CPU_THREADS       = rp_Hydro::N_THREADS_FOR_ADV_SOLN;
constexpr  unsigned int N_GPU_THREADS       = 0;
constexpr  unsigned int N_BLKS_PER_PACKET   = 0;
constexpr  unsigned int N_BLKS_PER_CPU_TURN = 1;

int main(int argc, char* argv[]) {
    // TODO: Add in error handling code

    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    orchestration::Logger::instantiate(rp_Simulation::LOG_FILENAME,
                                       GLOBAL_COMM, LOG_RANK);

    // Analogous to calling Grid_init
    loadGridConfiguration();
    orchestration::Grid::instantiate();

    // Analogous to calling IO_init
    orchestration::Io::instantiate(rp_Simulation::INTEGRAL_QUANTITIES_FILENAME,
                                   GLOBAL_COMM, IO_RANK);

    // Analogous to calling sim_init
    std::vector<std::string>  variableNames = sim::getVariableNames();

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    ProcessTimer  hydro{rp_Simulation::NAME + "_timings.dat", "MPI+OpenMP",
                        N_DIST_THREADS, 0, N_CPU_THREADS, N_GPU_THREADS,
                        N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN,
                        GLOBAL_COMM, TIMER_RANK};

    //----- MIMIC Grid_initDomain
    orchestration::Io&       io      = orchestration::Io::instance();
    orchestration::Grid&     grid    = orchestration::Grid::instance();
    orchestration::Logger&   logger  = orchestration::Logger::instance();

    Driver::dt      = rp_Simulation::DT_INIT;
    Driver::simTime = rp_Simulation::T_0;

    Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);
    Timer::stop("Set initial conditions");

    Timer::start("computeLocalIQ");
    io.computeLocalIntegralQuantities();
    Timer::stop("computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
//    grid.writePlotfile(rp_Simulation::NAME + "_plt_ICs", variableNames);
    Timer::stop("Reduce/Write");

    //----- MIMIC Driver_evolveFlash
    Timer::start(rp_Simulation::NAME + " simulation");

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
        if (rank == LEAD_RANK) {
            printf("Step n=%d / t=%.4e / dt=%.4e\n", nStep, Driver::simTime, Driver::dt);
        }

        //----- ADVANCE SOLUTION BASED ON HYDRODYNAMICS
        if (nStep > 1) {
            Timer::start("GC Fill");
            grid.fillGuardCells();
            Timer::stop("GC Fill");
        }

        //----- ADVANCE SOLUTION
        // Update unk data on interiors only
        double   tStart = MPI_Wtime();
#pragma omp parallel default(none) \
                     private(tileDesc) \
                     shared(grid, level, Driver::dt) \
                     num_threads(rp_Hydro::N_THREADS_FOR_ADV_SOLN)
        {
            // Create thread-private scratch buffers.
            // They will be reindexed as needed, but size needs to be correct.
            FArray4D   flX = FArray4D::buildScratchArray4D(
                                IntVect{LIST_NDIM(1,                1,            1)},
                                IntVect{LIST_NDIM(rp_Grid::NXB+K1D, rp_Grid::NYB, rp_Grid::NZB)},
                                NFLUXES);
            FArray4D   flY = FArray4D::buildScratchArray4D(
                                IntVect{LIST_NDIM(1,            1,                1)},
                                IntVect{LIST_NDIM(rp_Grid::NXB, rp_Grid::NYB+K2D, rp_Grid::NZB)},
                                NFLUXES);
            FArray4D   flZ = FArray4D::buildScratchArray4D(
                                IntVect{LIST_NDIM(1,            1,            1)},
                                IntVect{LIST_NDIM(rp_Grid::NXB, rp_Grid::NYB, rp_Grid::NZB+K3D)},
                                NFLUXES);
            FArray3D   auxC = FArray3D::buildScratchArray(
                                IntVect{LIST_NDIM(1  -K1D,          1  -K2D,          1  -K3D)},
                                IntVect{LIST_NDIM(rp_Grid::NXB+K1D, rp_Grid::NYB+K2D, rp_Grid::NZB+K3D)});

            for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
                tileDesc = ti->buildCurrentTile();

                const IntVect       lo = tileDesc->lo();
                const IntVect       hi = tileDesc->hi();
                FArray4D            U  = tileDesc->data();

                const IntVect       cLo = IntVect{LIST_NDIM(lo.I()-K1D,
                                                            lo.J()-K2D,
                                                            lo.K()-K3D)};
                flX.reindex(lo);
                flY.reindex(lo);
                flZ.reindex(lo);
                auxC.reindex(cLo);

                hy::computeFluxesHll(Driver::dt, lo, hi,
                                     tileDesc->deltas(),
                                     U, flX, flY, flZ, auxC);
                hy::updateSolutionHll(lo, hi, U, flX, flY, flZ);
                Eos::idealGammaDensIe(lo, hi, U);
            }
        }
        double       wtime_sec = MPI_Wtime() - tStart;
        Timer::start("Gather/Write");
        hydro.logTimestep(nStep, wtime_sec);
        Timer::stop("Gather/Write");

        Timer::start("computeLocalIQ");
        io.computeLocalIntegralQuantities();
        Timer::stop("computeLocalIQ");

        //----- OUTPUT RESULTS TO FILES
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

        ++nStep;
    }
    Timer::stop(rp_Simulation::NAME + " simulation");

    if (Driver::simTime >= rp_Simulation::T_MAX) {
        Logger::instance().log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile(rp_Simulation::NAME + "_plt_final",
                       variableNames);

    nStep = std::min(nStep, rp_Simulation::MAX_STEPS);

    //----- CLEAN-UP
    // The singletons are finalized automatically when the program is
    // terminating.

    return 0;
}

