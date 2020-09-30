#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Tile.h"
#include "Grid.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "Flash.h"
#include "constants.h"
#include "setInitialConditions.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"
#include "scaleEnergy.h"
#include "Analysis.h"

#ifdef USE_CUDA_BACKEND
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"
#endif

// Have make specify build-time constants for file headers
#include "buildInfo.h"

// No AMR.  Set level to 0-based AMReX coarse level
constexpr unsigned int   LEVEL = 0;

constexpr unsigned int   N_TRIALS = 5;

constexpr unsigned int   N_THREAD_TEAMS = 3;
constexpr unsigned int   MAX_THREADS = 7;
constexpr int            N_STREAMS = 32; 
constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 8589934592; 

void  setUp(void) {
    using namespace orchestration;

    RuntimeAction    setICs;
    setICs.name            = "SetICs";
    setICs.nInitialThreads = 4;
    setICs.teamType        = ThreadTeamDataType::BLOCK;
    setICs.nTilesPerPacket = 0;
    setICs.routine         = ActionRoutines::setInitialConditions_tile_cpu;

    Runtime::instance().executeCpuTasks("SetICs", setICs);
}

void  tearDown(const std::string& filename,
               const std::string& mode,
               const unsigned int nLoops,
               const int nThdTask1, 
               const int nThdTask2, 
               const int nThdTask3,
               const double walltime) {
    using namespace orchestration;

    int   rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // TODO: Don't hardcode level
        RealVect   deltas = Grid::instance().getDeltas(LEVEL);
        Real       dx = deltas[Axis::I];
        Real       dy = deltas[Axis::J];

        RuntimeAction    computeError;
        computeError.name            = "ComputeErrors";
        computeError.nInitialThreads = 3;
        computeError.teamType        = ThreadTeamDataType::BLOCK;
        computeError.nTilesPerPacket = 0;
        computeError.routine         = ActionRoutines::computeErrors_tile_cpu;

        double   L_inf_dens = 0.0;
        double   L_inf_ener = 0.0;
        double   meanAbsError = 0.0;
        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        Runtime::instance().executeCpuTasks("ComputeErrors", computeError);

        Analysis::densityErrors(&L_inf_dens, &meanAbsError);
        Analysis::energyErrors(&L_inf_ener, &meanAbsError);

        std::ofstream  fptr;
        fptr.open(filename, std::ios::out | std::ios::app);
        fptr << std::setprecision(15) 
             << mode << ","
             << nLoops << ","
             << (nThdTask1 < 0 ? "NaN" : std::to_string(nThdTask1)) << ","
             << (nThdTask2 < 0 ? "NaN" : std::to_string(nThdTask2)) << ","
             << (nThdTask3 < 0 ? "NaN" : std::to_string(nThdTask3)) << ","
             << NXB << "," << NYB << "," << NZB << ","
             << N_BLOCKS_X << ","
             << N_BLOCKS_Y << ","
             << N_BLOCKS_Z << ","
             << dx << "," << dy << ","
             << L_inf_dens << "," << L_inf_ener << ","
             << walltime << std::endl;
        fptr.close();
    }
}

int   main(int argc, char* argv[]) {
    using namespace orchestration;

    // Initialize simulation
    orchestration::Runtime::setNumberThreadTeams(N_THREAD_TEAMS);
    orchestration::Runtime::setMaxThreadsPerTeam(MAX_THREADS);
    orchestration::Runtime::setLogFilename("GatherDataCpp.log");

#ifdef USE_CUDA_BACKEND
    orchestration::CudaStreamManager::setMaxNumberStreams(N_STREAMS);
    orchestration::CudaMemoryManager::setBufferSize(MEMORY_POOL_SIZE_BYTES);
#endif

    orchestration::Runtime&   runtime = orchestration::Runtime::instance();

    Grid::instantiate();
    Grid&   grid = Grid::instance();
    grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu);

    // Setup logging of results
    std::string  fname("gatherDataCpp_");
    fname += std::to_string(NXB) + "_";
    fname += std::to_string(NYB) + "_";
    fname += std::to_string(NZB) + "_";
    fname += std::to_string(N_BLOCKS_X) + "_";
    fname += std::to_string(N_BLOCKS_Y) + "_";
    fname += std::to_string(N_BLOCKS_Z) + ".dat";

    // Write header to file
    std::ofstream  fptr;
    fptr.open(fname, std::ios::out);
	fptr << "# Git repository," << PROJECT_GIT_REPO_NAME << std::endl;
    fptr << "# Git Commit," << PROJECT_GIT_REPO_VER << std::endl;
    fptr << "# AMReX version," << amrex::Version() << std::endl;
    fptr << "# C++ compiler," << CXX_COMPILER << std::endl;
//    fptr << "# C++ compiler version," << CXX_COMPILER_VERSION << std::endl;
	fptr << "# Build date," << BUILD_DATETIME << std::endl;
	fptr << "# Hostname, " << HOSTNAME << std::endl;
//    fptr << "# Host information," << MACHINE_INFO << std::endl;
    fptr << "# MPI_Wtick," << MPI_Wtick() << ",sec" << std::endl;
    fptr << "pmode,n_loops,n_thd_task1,n_thd_task2,n_thd_task3,"
         << "NXB,NYB,NZB,N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z,"
         << "dx,dy,Linf_density,Linf_energy,Walltime_sec\n";
    fptr.close();

    std::string   testName  = "NXB=" + std::to_string(NXB) + " / ";
                  testName += "NYB=" + std::to_string(NYB) + " / ";
                  testName += "NZB=" + std::to_string(NZB) + " / ";
                  testName += "N Blocks X=" + std::to_string(N_BLOCKS_X) + " / ";
                  testName += "N Blocks Y=" + std::to_string(N_BLOCKS_Y) + " / ";
                  testName += "N Blocks Z=" + std::to_string(N_BLOCKS_Z);
    Logger::instance().log("[Simulation] Start " + testName);

    for (unsigned int j=0; j<N_TRIALS; ++j) {
        Logger::instance().log("[Simulation] Start trial " + std::to_string(j+1));

        /***** SERIAL TEST - Three iteration loops  *****/
        Logger::instance().log("[Simulation] Start 3 loop Serial Test");
        setUp();
        double tStart = MPI_Wtime(); 
        std::unique_ptr<Tile>    dataItem{};
        for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next()) {
            dataItem = ti->buildCurrentTile();
            ActionRoutines::computeLaplacianDensity_tile_cpu(0, dataItem.get());
        }

        for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next()) {
            dataItem = ti->buildCurrentTile();
            ActionRoutines::computeLaplacianEnergy_tile_cpu(0, dataItem.get());
        }

        for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next()) {
            dataItem = ti->buildCurrentTile();
            ActionRoutines::scaleEnergy_tile_cpu(0, dataItem.get());
        }
        double tWalltime = MPI_Wtime() - tStart;
        tearDown(fname, "Serial", 3, -1, -1, -1, tWalltime);
        Logger::instance().log("[Simulation] End 3 loop Serial Test");

        /***** SERIAL TEST - Single iteration loop  *****/
        Logger::instance().log("[Simulation] Start Single Loop Serial Test");
        setUp();
        tStart = MPI_Wtime(); 
        for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next()) {
            dataItem = ti->buildCurrentTile();
            ActionRoutines::computeLaplacianDensity_tile_cpu(0, dataItem.get());
            ActionRoutines::computeLaplacianEnergy_tile_cpu(0, dataItem.get());
            ActionRoutines::scaleEnergy_tile_cpu(0, dataItem.get());
        }
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "Serial", 1, -1, -1, -1, tWalltime);
        Logger::instance().log("[Simulation] End Single Loop Serial Test");

        /***** RUNTIME TEST - Serialized version *****/
        Logger::instance().log("[Simulation] Start 3 Loop Serialized Runtime Test");
        RuntimeAction    computeLaplacianDensity;
        RuntimeAction    computeLaplacianEnergy;
        RuntimeAction    scaleEnergy;

        computeLaplacianDensity.name            = "LaplacianDensity";
        computeLaplacianDensity.nInitialThreads = 1;
        computeLaplacianDensity.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianDensity.nTilesPerPacket = 0;
        computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

        computeLaplacianEnergy.name            = "LaplacianEnergy";
        computeLaplacianEnergy.nInitialThreads = 1;
        computeLaplacianEnergy.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianEnergy.nTilesPerPacket = 0;
        computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_tile_cpu;

        scaleEnergy.name                       = "scaleEnergy";
        scaleEnergy.nInitialThreads            = 1;
        scaleEnergy.teamType                   = ThreadTeamDataType::BLOCK;
        scaleEnergy.nTilesPerPacket            = 0;
        scaleEnergy.routine                    = ActionRoutines::scaleEnergy_tile_cpu;

        setUp();

        tStart = MPI_Wtime(); 
        runtime.executeCpuTasks("LapDens", computeLaplacianDensity);
        tWalltime = MPI_Wtime() - tStart; 

        tStart = MPI_Wtime(); 
        runtime.executeCpuTasks("LapEner", computeLaplacianEnergy);
        tWalltime += MPI_Wtime() - tStart; 

        tStart = MPI_Wtime(); 
        runtime.executeCpuTasks("scEner",  scaleEnergy);
        tWalltime += MPI_Wtime() - tStart; 

        tearDown(fname, "Runtime", 3, 1, 0, 0, tWalltime);
        Logger::instance().log("[Simulation] End 3 Loop Serialized Runtime Test");

        /***** RUNTIME TEST - Try different thread combinations *****/
#if defined(USE_CUDA_BACKEND)
        Logger::instance().log("[Simulation] Start CPU/GPU/Post-GPU Runtime Test");
        setUp();

        computeLaplacianDensity.nInitialThreads = 2;

        computeLaplacianEnergy.nInitialThreads  = 5;
        computeLaplacianEnergy.teamType         = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianEnergy.nTilesPerPacket  = 4;
        computeLaplacianEnergy.routine          = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

        scaleEnergy.nInitialThreads             = 0;

        tStart = MPI_Wtime(); 
        runtime.executeTasks_FullPacket("FullPacket",
                                        computeLaplacianDensity,
                                        computeLaplacianEnergy,
                                        scaleEnergy);
        tWalltime = MPI_Wtime() - tStart; 

        tearDown(fname, "Runtime", 1, 2, 5, 0, tWalltime);
        Logger::instance().log("[Simulation] End CPU/GPU/Post-GPU Runtime Test");
#endif

        Logger::instance().log("[Simulation] End trial " + std::to_string(j+1));
    }

    grid.destroyDomain();

    Logger::instance().log("[Simulation] End " + testName);

    return 0;
}

