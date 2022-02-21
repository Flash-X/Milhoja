#include <cstdio>
#include <string>

#include <mpi.h>

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Logger.h>

#include "Sedov.h"
#include "RuntimeParameters.h"
#include "Io.h"
#include "Eos.h"
#include "Hydro.h"
#include "Timer.h"
#include "Driver.h"
#include "Simulation.h"
#include "ProcessTimer.h"

void    Driver::executeSimulation(void) {
    constexpr  int          TIMER_RANK          = LEAD_RANK;
    constexpr  unsigned int N_DIST_THREADS      = 0;
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

    Driver::dt      = RPs.getReal("Simulation", "dtInit");
    Driver::simTime = RPs.getReal("Simulation", "T_0"); 
    unsigned int   nCpuThreads{RPs.getUnsignedInt("Hydro", "nThreadsForAdvanceSolution")};

    ProcessTimer  hydro{"sedov_timings.dat", "MPI+OpenMP",
                        N_DIST_THREADS, 0, nCpuThreads, N_GPU_THREADS,
                        N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN,
                        GLOBAL_COMM, TIMER_RANK};

    Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu);
    Timer::stop("Set initial conditions");

    Timer::start("computeLocalIQ");
    io.computeLocalIntegralQuantities();
    Timer::stop("computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
//    grid.writePlotfile("sedov_plt_ICs", variableNames);
    Timer::stop("Reduce/Write");

    Timer::start("sedov simulation");

    int                NXB{RPs.getInt("Grid", "NXB")};
    int                NYB{RPs.getInt("Grid", "NYB")};
    int                NZB{RPs.getInt("Grid", "NZB")};
    milhoja::IntVect   fl_lo{LIST_NDIM(1, 1, 1)};
    milhoja::IntVect   flX_hi{LIST_NDIM(NXB+MILHOJA_K1D, NYB,             NZB)};
    milhoja::IntVect   flY_hi{LIST_NDIM(NXB,             NYB+MILHOJA_K2D, NZB)};
    milhoja::IntVect   flZ_hi{LIST_NDIM(NXB,             NYB,             NZB+MILHOJA_K3D)};
    milhoja::IntVect   auxC_lo{LIST_NDIM(1  -MILHOJA_K1D, 1  -MILHOJA_K2D, 1  -MILHOJA_K3D)};
    milhoja::IntVect   auxC_hi{LIST_NDIM(NXB+MILHOJA_K1D, NYB+MILHOJA_K2D, NZB+MILHOJA_K3D)};

    unsigned int                     level{0};
    std::shared_ptr<milhoja::Tile>   tileDesc{};
    unsigned int                     nStep{1};
    unsigned int                     maxSteps{RPs.getUnsignedInt("Simulation", "maxSteps")};
    milhoja::Real                    tMax{RPs.getReal("Simulation", "tMax")};
    milhoja::Real                    dtAfter{RPs.getReal("Driver", "dtAfter")};
    unsigned int                     writeEveryNSteps{RPs.getUnsignedInt("Driver", "writeEveryNSteps")};
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

        //----- ADVANCE SOLUTION
        // Update unk data on interiors only
        double   tStart = MPI_Wtime();
#pragma omp parallel default(none) \
                     private(tileDesc) \
                     shared(grid, level, Driver::dt, fl_lo, flX_hi, flY_hi, flZ_hi, auxC_lo, auxC_hi) \
                     num_threads(nCpuThreads)
        {
            // Create thread-private scratch buffers.
            // They will be reindexed as needed, but size needs to be correct.
            milhoja::FArray4D   flX  = milhoja::FArray4D::buildScratchArray4D(fl_lo, flX_hi, NFLUXES);
            milhoja::FArray4D   flY  = milhoja::FArray4D::buildScratchArray4D(fl_lo, flY_hi, NFLUXES);
            milhoja::FArray4D   flZ  = milhoja::FArray4D::buildScratchArray4D(fl_lo, flZ_hi, NFLUXES);
            milhoja::FArray3D   auxC = milhoja::FArray3D::buildScratchArray(auxC_lo, auxC_hi);

            for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
                tileDesc = ti->buildCurrentTile();

                const milhoja::IntVect       lo = tileDesc->lo();
                const milhoja::IntVect       hi = tileDesc->hi();
                milhoja::FArray4D            U  = tileDesc->data();

                const milhoja::IntVect       cLo = milhoja::IntVect{LIST_NDIM(lo.I()-MILHOJA_K1D,
                                                                              lo.J()-MILHOJA_K2D,
                                                                              lo.K()-MILHOJA_K3D)};
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
    grid.writePlotfile("sedov_plt_final",
                       variableNames);

    nStep = std::min(nStep, maxSteps);

    grid.destroyDomain();
}

