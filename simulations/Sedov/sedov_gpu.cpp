#include <cstdio>
#include <string>

#include <mpi.h>

#include "Io.h"
#include "Eos.h"
#include "Hydro.h"
#include "Driver.h"
#include "Simulation.h"

#include "Grid_REAL.h"
#include "Grid.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "errorEstBlank.h"

#include "Flash_par.h"

int main(int argc, char* argv[]) {
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
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    //----- MIMIC Grid_initDomain
    orchestration::Io&       io      = orchestration::Io::instance();
    orchestration::Grid&     grid    = orchestration::Grid::instance();
    orchestration::Logger&   logger  = orchestration::Logger::instance();
    orchestration::Runtime&  runtime = orchestration::Runtime::instance();

    Driver::dt      = rp_Simulation::DT_INIT;
    Driver::simTime = rp_Simulation::T_0;

    logger.log("[Simulation] Generate mesh and set initial conditions");
    grid.initDomain(Simulation::setInitialConditions_tile_cpu,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstBlank);

    //----- OUTPUT RESULTS TO FILES
    // This only makes sense if the iteration is over LEAF blocks.
    RuntimeAction     computeIntQuantitiesByBlk;
    computeIntQuantitiesByBlk.name            = "Compute Integral Quantities";
    computeIntQuantitiesByBlk.nInitialThreads = rp_Io::N_THREADS_FOR_INT_QUANTITIES;
    computeIntQuantitiesByBlk.teamType        = ThreadTeamDataType::BLOCK;
    computeIntQuantitiesByBlk.nTilesPerPacket = 0;
    computeIntQuantitiesByBlk.routine         
        = ActionRoutines::Io_computeIntegralQuantitiesByBlock_tile_cpu;

    // TODO: Shouldn't this be done through the IO unit?
    grid.writePlotfile(rp_Simulation::NAME + "_plt_ICs");

    // Compute local integral quantities
    runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
    // Compute global integral quantities via DATA MOVEMENT
    io.reduceToGlobalIntegralQuantities();
    io.writeIntegralQuantities(Driver::simTime);

    //----- MIMIC Driver_evolveFlash
    RuntimeAction     hydroAdvance;
    hydroAdvance.name            = "Advance Hydro Solution";
    hydroAdvance.nInitialThreads = rp_Hydro::N_THREADS_FOR_ADV_SOLN;
    hydroAdvance.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    hydroAdvance.nTilesPerPacket = rp_Hydro::N_BLOCKS_PER_PACKET_FOR_ADV_SOLN;
    hydroAdvance.routine         = Hydro::advanceSolutionHll_packet_oacc_summit_3;

//    RuntimeAction     applyEos;
//    applyEos.name            = "Ideal Gamma Dens/Ie";
//    applyEos.nInitialThreads = 2;
//    applyEos.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
//    applyEos.nTilesPerPacket = 80;
//    applyEos.routine         = Eos::idealGammaDensIe_packet_oacc_summit;

    logger.log("[Simulation] " + rp_Simulation::NAME + " simulation started");

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
            grid.fillGuardCells();
        }
        runtime.executeGpuTasks("Advance Hydro Solution", hydroAdvance);
//        runtime.executeGpuTasks("Apply Eos", applyEos);

        if ((nStep % rp_Driver::WRITE_EVERY_N_STEPS) == 0) {
            grid.writePlotfile(rp_Simulation::NAME + "_plt_" + std::to_string(nStep));
        }

        //----- OUTPUT RESULTS TO FILES
        // Compute local integral quantities
        // TODO: This should be run as a CPU-based pipeline extension
        //       to the physics action bundle.
        runtime.executeCpuTasks("IntegralQ", computeIntQuantitiesByBlk);
        io.reduceToGlobalIntegralQuantities();
        io.writeIntegralQuantities(Driver::simTime);

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
    logger.log("[Simulation] " + rp_Simulation::NAME + " simulation terminated");
    if (Driver::simTime >= rp_Simulation::T_MAX) {
        Logger::instance().log("[Simulation] Reached max SimTime");
    }
    grid.writePlotfile(rp_Simulation::NAME + "_plt_final");

    nStep = std::min(nStep, rp_Simulation::MAX_STEPS);

    //----- CLEAN-UP
    // The singletons are finalized automatically when the program is
    // terminating.

    return 0;
}

