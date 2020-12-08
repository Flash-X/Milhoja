#include <cstdio>
#include <string>

#include <mpi.h>

#include "IO.h"
#include "Hydro.h"
#include "Driver.h"
#include "Simulation.h"
#include "Orchestration.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

// TODO: This should be managed and correct assignment confirmed at the same
// time that we make certain that the Real type specified in this repo matches
// the Real used in the Grid backend.
constexpr  int         ORCH_REAL                    = MPI_DOUBLE_PRECISION;
constexpr  int         GLOBAL_COMM                  = MPI_COMM_WORLD;
const      std::string SIMULATION_NAME              = "sedov";
const      std::string LOG_FILENAME                 = SIMULATION_NAME + ".log";
const      std::string INTEGRAL_QUANTITIES_FILENAME = SIMULATION_NAME + ".dat";

int main(int argc, char* argv[]) {
    // TODO: Add in error handling code

    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    orchestration::Logger::instantiate(LOG_FILENAME);

    // Analogous to calling Orchestration_init
    orchestration::Runtime::instantiate(orch::nThreadTeams, 
                                        Orchestration::nThreadsPerTeam,
                                        orch::nStreams,
                                        orch::memoryPoolSizeBytes);

    // Analogous to calling Grid_init
    orchestration::Grid::instantiate();

    // Analogous to calling IO_init
    IO::initialize(INTEGRAL_QUANTITIES_FILENAME);

    int  rank = 0;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    //----- MIMIC Grid_initDomain
    orchestration::Grid&     grid    = orchestration::Grid::instance();
    orchestration::Logger&   logger  = orchestration::Logger::instance();
    orchestration::Runtime&  runtime = orchestration::Runtime::instance();

    Driver::dt      = Simulation::dtInit;
    Driver::simTime = Simulation::t_0;

    logger.log("[Simulation] Generate mesh and set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    Simulation::errorEstBlank);

    //----- OUTPUT RESULTS TO FILES
    // This only makes sense if the iteration is over LEAF blocks.
    RuntimeAction     computeBlockIntQuantities;
    computeBlockIntQuantities.name            = "Compute Integral Quantities";
    computeBlockIntQuantities.nInitialThreads = 2;
    computeBlockIntQuantities.teamType        = ThreadTeamDataType::BLOCK;
    computeBlockIntQuantities.nTilesPerPacket = 0;
    computeBlockIntQuantities.routine         = IO::computeBlockIntegralQuantities_tile_cpu;

    grid.writePlotfile(SIMULATION_NAME + "_ICs");

    // Compute local integral quantities
    runtime.executeCpuTasks("IntegralQ", computeBlockIntQuantities);
    IO::computeLocalIntegralQuantities();

    // Compute  global integral quantities
    int err = MPI_Reduce((void*)IO::localIntegralQuantities,
                         (void*)IO::globalIntegralQuantities,
                         IO::nIntegralQuantities, ORCH_REAL, MPI_SUM,
                         MASTER_PE, MPI_COMM_WORLD);
    IO::writeIntegralQuantities(Driver::simTime);

    //----- MIMIC Driver_evolveFlash
    RuntimeAction     hydroAdvance;
    hydroAdvance.name            = "Advance Hydro Solution";
    hydroAdvance.nInitialThreads = 2;
    hydroAdvance.teamType        = ThreadTeamDataType::BLOCK;
    hydroAdvance.nTilesPerPacket = 0;
    hydroAdvance.routine         = Hydro::advanceSolution_tile_cpu;

    logger.log("[Simulation] " + SIMULATION_NAME + " simulation started");

    unsigned int   nStep   = 1;
    while ((nStep <= Simulation::maxSteps) && (Driver::simTime < Simulation::t_max)) {
        //----- ADVANCE TIME
        // Don't let simulation time exceed maximum simulation time
        if ((Driver::simTime + Driver::dt) > Simulation::t_max) {
            Real   origDt = Driver::dt;
            Driver::dt = (Simulation::t_max - Driver::simTime);
            Driver::simTime = Simulation::t_max;
            logger.log(  "[Driver] Shortened dt from " + std::to_string(origDt)
                       + " to " + std::to_string(Driver::dt)
                       + " so that tmax=" + std::to_string(Simulation::t_max)
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
            grid.fillGuardCells();
        }
        runtime.executeCpuTasks("Advance Hydro Solution", hydroAdvance);

        if ((nStep % dr::writeEveryNSteps) == 0) {
            grid.writePlotfile(SIMULATION_NAME + "_" + std::to_string(nStep));
        }

        //----- OUTPUT RESULTS TO FILES
        // Compute local integral quantities
        // TODO: This should be run as a CPU-based pipeline extension
        //       to the physics action bundle.
        runtime.executeCpuTasks("IntegralQ", computeBlockIntQuantities);
        IO::computeLocalIntegralQuantities();

        // Compute  global integral quantities
        err = MPI_Reduce((void*)IO::localIntegralQuantities,
                         (void*)IO::globalIntegralQuantities,
                         IO::nIntegralQuantities, ORCH_REAL, MPI_SUM,
                         MASTER_PE, MPI_COMM_WORLD);
        IO::writeIntegralQuantities(Driver::simTime);

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
        // Simulation::dtInit.  There after, it allows for 5.0e-5.  Therefore,
        // we mimic that dt sequence here so that we can directly compare
        // results.
        Driver::dt = dr::dtAfter;

        ++nStep;
    }
    logger.log("[Simulation] " + SIMULATION_NAME + " simulation terminated");
    if (Driver::simTime >= Simulation::t_max) {
        Logger::instance().log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile(SIMULATION_NAME + "_final");

    nStep = std::min(nStep, Simulation::maxSteps);

    //----- CLEAN-UP
    // The singletons are finalized automatically when the program is
    // terminating.

    IO::finalize();

    return 0;
}

