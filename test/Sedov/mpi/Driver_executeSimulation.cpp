#include <cstdio>
#include <string>

#include <mpi.h>

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Logger.h>
#include <Milhoja_test.h>

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
    constexpr  unsigned int N_CPU_THREADS       = 1;
    constexpr  unsigned int N_GPU_THREADS       = 0;
    constexpr  unsigned int N_BLKS_PER_PACKET   = 0;
    constexpr  unsigned int N_BLKS_PER_CPU_TURN = 1;

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    ProcessTimer  hydro{"sedov_timings.dat", "MPI",
                        N_DIST_THREADS, 0, N_CPU_THREADS, N_GPU_THREADS,
                        N_BLKS_PER_PACKET, N_BLKS_PER_CPU_TURN,
                        GLOBAL_COMM, TIMER_RANK};

    // Analogous to calling sim_init
    std::vector<std::string>  variableNames = sim::getVariableNames();

    Io&                      io      = Io::instance();
    RuntimeParameters&       RPs     = RuntimeParameters::instance();
    milhoja::Grid&           grid    = milhoja::Grid::instance();
    milhoja::Logger&         logger  = milhoja::Logger::instance();

    Driver::dt      = RPs.getReal("Simulation", "dtInit");
    Driver::simTime = RPs.getReal("Simulation", "T_0"); 

    Timer::start("Set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_noRuntime);
    Timer::stop("Set initial conditions");

    Timer::start("computeLocalIQ");
    io.computeLocalIntegralQuantities();
    Timer::stop("computeLocalIQ");

    //----- OUTPUT RESULTS TO FILES
    // Compute global integral quantities via DATA MOVEMENT
    Timer::start("Reduce/Write");
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);
//   grid.writePlotfile("sedov_plt_ICs", variableNames);
    Timer::stop("Reduce/Write");

    //----- MIMIC Driver_evolveFlash
    // Create scratch buffers.
    // They will be reindexed as needed, but size needs to be correct.
    int    NXB{MILHOJA_TEST_NXB};
    int    NYB{MILHOJA_TEST_NYB};
    int    NZB{MILHOJA_TEST_NZB};
    milhoja::FArray3D   auxC = milhoja::FArray3D::buildScratchArray(
                            milhoja::IntVect{LIST_NDIM(1  -MILHOJA_K1D, 1  -MILHOJA_K2D, 1  -MILHOJA_K3D)},
                            milhoja::IntVect{LIST_NDIM(NXB+MILHOJA_K1D, NYB+MILHOJA_K2D, NZB+MILHOJA_K3D)});

    Timer::start("sedov simulation");

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
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();

            const milhoja::IntVect       lo  = tileDesc->lo();
            const milhoja::IntVect       hi  = tileDesc->hi();
            milhoja::FArray4D            U   = tileDesc->data();
            milhoja::FArray4D            flX = tileDesc->fluxData(milhoja::Axis::I);
#if MILHOJA_NDIM >= 2
            milhoja::FArray4D            flY = tileDesc->fluxData(milhoja::Axis::J);
#endif
#if MILHOJA_NDIM == 3
            milhoja::FArray4D            flZ = tileDesc->fluxData(milhoja::Axis::K);
#endif

            const milhoja::IntVect       cLo = milhoja::IntVect{LIST_NDIM(lo.I()-MILHOJA_K1D,
                                                                          lo.J()-MILHOJA_K2D,
                                                                          lo.K()-MILHOJA_K3D)};
            auxC.reindex(cLo);

            hy::computeFluxesHll(Driver::dt, lo, hi,
                                 tileDesc->deltas(),
                                 U, LIST_NDIM(flX, flY, flZ), auxC);
            hy::updateSolutionHll(lo, hi, U, LIST_NDIM(flX, flY, flZ));
            Eos::idealGammaDensIe(lo, hi, U);
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
    grid.writePlotfile("sedov_plt_final", variableNames);
    
    nStep = std::min(nStep, maxSteps);
    
    grid.destroyDomain();
}

