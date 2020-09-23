#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "Grid.h"
#include "RuntimeAction.h"
#include "CudaRuntime.h"
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"

#include "Flash.h"
#include "constants.h"
#include "setInitialConditions_block.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"
#include "scaleEnergy.h"
#include "Analysis.h"

// Have make specify build-time constants for file headers
#include "buildInfo.h"

// No AMR.  Set level to 0-based AMReX coarse level
constexpr unsigned int   LEVEL = 0;

constexpr unsigned int   N_TRIALS = 5;
// It appears that OpenACC on Summit with PGI has max 32 asynchronous
// queues.  If you assign more CUDA streams to queues with OpenACC, then
// these streams just roll over and the last 32 CUDA streams will be the
// only streams mapped to queues.
constexpr int            N_STREAMS = 32; 
constexpr unsigned int   N_THREAD_TEAMS = 3;
constexpr unsigned int   MAX_THREADS = 7;
constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 4294967296; 

void  setUp(void) {
    using namespace orchestration;

    RuntimeAction    setICs;
    setICs.nInitialThreads = 6;
    setICs.teamType        = ThreadTeamDataType::BLOCK;
    setICs.nTilesPerPacket = 0;
    setICs.routine         = Simulation::setInitialConditions_block;

    CudaRuntime::instance().executeCpuTasks("setICs", setICs);
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
        Real       dx = deltas.I();
        Real       dy = deltas.J();

        RuntimeAction    computeError_block;
        computeError_block.nInitialThreads = 6;
        computeError_block.teamType        = ThreadTeamDataType::BLOCK;
        computeError_block.nTilesPerPacket = 0;
        computeError_block.routine         = Analysis::computeErrors_block;

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        CudaRuntime::instance().executeCpuTasks("ComputeErrors", computeError_block);

        double   L_inf_dens = 0.0;
        double   L_inf_ener = 0.0;
        double   meanAbsError = 0.0;
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
    orchestration::CudaRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    orchestration::CudaRuntime::setMaxThreadsPerTeam(MAX_THREADS);
    orchestration::CudaRuntime::setLogFilename("GatherDataCuda.log");

    orchestration::CudaRuntime&   runtime = orchestration::CudaRuntime::instance();
    std::cout << "\n";
    std::cout << "----------------------------------------------------------\n";
    runtime.printGpuInformation();
    std::cout << "----------------------------------------------------------\n";
    std::cout << std::endl;

    orchestration::CudaStreamManager::setMaxNumberStreams(N_STREAMS);
    orchestration::CudaMemoryManager::setBufferSize(MEMORY_POOL_SIZE_BYTES);

    Grid::instantiate();
    Grid&   grid = Grid::instance();
    grid.initDomain(Simulation::setInitialConditions_block);

    // Setup logging of results
    std::string  fname("gatherDataCudaOaccCpp_");
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

    for (unsigned int j=0; j<N_TRIALS; ++j) {
        /***** SERIAL TEST - Single iteration loop  *****/
        setUp();
        double  tStart = MPI_Wtime(); 
        std::unique_ptr<DataItem>    dataItem{};
        for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next()) {
            dataItem = ti->buildCurrentTile();
            ActionRoutines::computeLaplacianDensity_tile_cpu(0, dataItem.get());
            ActionRoutines::computeLaplacianEnergy_tile_cpu(0, dataItem.get());
            ActionRoutines::scaleEnergy_tile_cpu(0, dataItem.get());
        }
        double  tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "Serial", 1, -1, -1, -1, tWalltime);

        /***** RUNTIME TEST - Run kernels serially in CPU with thread team *****/
        RuntimeAction    computeLaplacianDensity;
        computeLaplacianDensity.nInitialThreads = 6;
        computeLaplacianDensity.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianDensity.nTilesPerPacket = 0;
        computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

        RuntimeAction    computeLaplacianEnergy;
        computeLaplacianEnergy.nInitialThreads = 6;
        computeLaplacianEnergy.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianEnergy.nTilesPerPacket = 0;
        computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_tile_cpu;

        RuntimeAction    scaleEnergy;
        scaleEnergy.nInitialThreads = 6;
        scaleEnergy.teamType        = ThreadTeamDataType::BLOCK;
        scaleEnergy.nTilesPerPacket = 0;
        scaleEnergy.routine         = ActionRoutines::scaleEnergy_tile_cpu;

        setUp();
        tStart = MPI_Wtime(); 
        runtime.executeCpuTasks("Density", computeLaplacianDensity);
        runtime.executeCpuTasks("Energy",  computeLaplacianEnergy);
        runtime.executeCpuTasks("Scale",   scaleEnergy);
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "RuntimeCpu", 3, 6, 6, 6, tWalltime);

        /***** RUNTIME TEST - Run kernels serially in GPU with thread team *****/
        computeLaplacianDensity.nInitialThreads = 6;
        computeLaplacianDensity.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianDensity.nTilesPerPacket = 1;
        computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

        computeLaplacianEnergy.nInitialThreads = 6;
        computeLaplacianEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianEnergy.nTilesPerPacket = 1;
        computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

        scaleEnergy.nInitialThreads = 6;
        scaleEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        scaleEnergy.nTilesPerPacket = 1;
        scaleEnergy.routine         = ActionRoutines::scaleEnergy_packet_oacc_summit;

        setUp();
        tStart = MPI_Wtime(); 
        runtime.executeGpuTasks("Density", computeLaplacianDensity);
        runtime.executeGpuTasks("Energy",  computeLaplacianEnergy);
        runtime.executeGpuTasks("Scale",   scaleEnergy);
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "RuntimeGpu", 3, 6, 6, 6, tWalltime);

        /***** RUNTIME TEST - Serialized version *****/
        computeLaplacianDensity.nInitialThreads = 2;
        computeLaplacianDensity.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianDensity.nTilesPerPacket = 0;
        computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

        computeLaplacianEnergy.nInitialThreads = 5;
        computeLaplacianEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianEnergy.nTilesPerPacket = 1;
        computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

        scaleEnergy.nInitialThreads = 0;
        scaleEnergy.teamType        = ThreadTeamDataType::BLOCK;
        scaleEnergy.nTilesPerPacket = 0;
        scaleEnergy.routine         = ActionRoutines::scaleEnergy_tile_cpu;

        setUp();
        tStart = MPI_Wtime(); 
        runtime.executeTasks_FullPacket("FullPacket",
                                        computeLaplacianDensity,
                                        computeLaplacianEnergy,
                                        scaleEnergy);
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "Runtime", 1, 2, 5, 0, tWalltime);

    }

    grid.destroyDomain();

    return 0;
}

