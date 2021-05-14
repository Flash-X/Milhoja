#include <cstdio>
#include <string>

#include <mpi.h>

#include "Io.h"
#include "Eos.h"
#include "Hydro.h"
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
constexpr  unsigned int N_CPU_THREADS       = rp_Hydro::N_THREADS_FOR_ADV_SOLN;
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

    // Analogous to calling IO_init
    orchestration::Io::instantiate(rp_Simulation::INTEGRAL_QUANTITIES_FILENAME);

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    ProcessTimer  hydro{rp_Simulation::NAME + "_timings.dat", "MPI+OpenMP",
                        N_DIST_THREADS, 0, N_CPU_THREADS, N_GPU_THREADS,
                        N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN};

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
    orchestration::Timer::stop("Set initial conditions");

    orchestration::Timer::start("computeLocalIQ");
    io.computeLocalIntegralQuantities();
    orchestration::Timer::stop("computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    orchestration::Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
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
        double   tStart = MPI_Wtime();
#pragma omp parallel default(none) \
                     private(tileDesc) \
                     shared(grid, level, Driver::dt) \
                     num_threads(rp_Hydro::N_THREADS_FOR_ADV_SOLN)
        {
            // Create thread-private scratch buffers.
            // They will be reindexed as needed, but size needs to be correct.
            FArray4D   flX = FArray4D::buildScratchArray4D(
                                IntVect{LIST_NDIM(1,         1,   1)},
                                IntVect{LIST_NDIM(NXB+K1D, NYB, NZB)},
                                NFLUXES);
            FArray4D   flY = FArray4D::buildScratchArray4D(
                                IntVect{LIST_NDIM(1,         1,   1)},
                                IntVect{LIST_NDIM(NXB, NYB+K2D, NZB)},
                                NFLUXES);
            FArray4D   flZ = FArray4D::buildScratchArray4D(
                                IntVect{LIST_NDIM(1,     1,       1)},
                                IntVect{LIST_NDIM(NXB, NYB, NZB+K3D)},
                                NFLUXES);
            FArray3D   auxC = FArray3D::buildScratchArray(
                                IntVect{LIST_NDIM(1  -K1D, 1  -K2D, 1  -K3D)},
                                IntVect{LIST_NDIM(NXB+K1D, NYB+K2D, NZB+K3D)});

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
        orchestration::Timer::start("Gather/Write");
        hydro.logTimestep(nStep, wtime_sec);
        orchestration::Timer::stop("Gather/Write");

        orchestration::Timer::start("computeLocalIQ");
        io.computeLocalIntegralQuantities();
        orchestration::Timer::stop("computeLocalIQ");

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
//    grid.writePlotfile(rp_Simulation::NAME + "_plt_final");

    nStep = std::min(nStep, rp_Simulation::MAX_STEPS);

    //----- CLEAN-UP
    // The singletons are finalized automatically when the program is
    // terminating.

    return 0;
}
