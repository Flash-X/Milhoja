#include <cstdio>

#include "Hydro.h"
#include "Driver.h"
#include "Simulation.h"
#include "Orchestration.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

int main(int argc, char* argv[]) {
    // TODO: Add in error handling code

    //----- MIMIC Driver_init
    // Analogous to calling Log_init
    orchestration::Logger::instantiate("sedov.log");

    // Analogous to calling Orchestration_init
    orchestration::Runtime::instantiate(orch::nThreadTeams, 
                                        orch::nThreadsPerTeam,
                                        orch::nStreams,
                                        orch::memoryPoolSizeBytes);

    // Analogous to calling Grid_init
    orchestration::Grid::instantiate();

    //----- MIMIC Grid_initDomain
    orchestration::Grid&     grid   = orchestration::Grid::instance();
    orchestration::Logger&   logger = orchestration::Logger::instance();

    logger.log("[Simulation] Generate mesh and set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    Simulation::errorEstBlank);
    grid.writePlotfile("sedov_ICs");

    //----- MIMIC Driver_evolveFlash
    orchestration::Runtime&   runtime = orchestration::Runtime::instance();

    RuntimeAction     hydroAdvance;
    hydroAdvance.name            = "Advance Hydro Solution";
    hydroAdvance.nInitialThreads = 2;
    hydroAdvance.teamType        = ThreadTeamDataType::BLOCK;
    hydroAdvance.nTilesPerPacket = 0;
    hydroAdvance.routine         = Hydro::advanceSolution_tile_cpu;

    Real           simTime    = Simulation::t_0;
    unsigned int   nStep      = 1;
    logger.log("[Simulation] Sedov simulation started");

    Driver::dt = Simulation::dtInit;
    while ((nStep <= Simulation::maxSteps) && (simTime < Simulation::t_max)) {
        //----- ADVANCE TIME
        // Don't let simulation time exceed maximum simulation time
        if ((simTime + Driver::dt) > Simulation::t_max) {
            Real   origDt = Driver::dt;
            Driver::dt = (Simulation::t_max - simTime);
            simTime = Simulation::t_max;
            logger.log(  "[Driver] Shortened dt from " + std::to_string(origDt)
                       + " to " + std::to_string(Driver::dt)
                       + " so that tmax=" + std::to_string(Simulation::t_max)
                       + " is not exceeded");
        } else {
            simTime += Driver::dt;
        }
        // TODO: Log as well
        printf("[Driver] Step n=%d / t=%.4e / dt=%.4e\n", nStep, simTime, Driver::dt);

        //----- ADVANCE SOLUTION BASED ON HYDRODYNAMICS
        if (nStep > 1) {
            grid.fillGuardCells();
        }
        runtime.executeCpuTasks("Advance Hydro Solution", hydroAdvance);

        if ((nStep % dr::writeEveryNSteps) == 0) {
            grid.writePlotfile("sedov_" + std::to_string(nStep));
        }

        //----- UPDATE GRID IF REQUIRED
        // We are running in pseudo-UG for now and can therefore skip this

        //----- COMPUTE dt FOR NEXT STEP
        // NOTE: The per-block computation of dt could be a follow up task run
        //       on the CPU.  The AllReduce that follows should appear here
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
    logger.log("[Simulation] Sedov simulation terminated");
    if (simTime >= Simulation::t_max) {
        Logger::instance().log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile("sedov_final");

    nStep = std::min(nStep, Simulation::maxSteps);

    return 0;
}

