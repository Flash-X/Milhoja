#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "Tile.h"
#include "Grid.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "Flash.h"
#include "constants.h"
#include "setInitialConditions.h"
#include "sleepyGoByeBye.h"

// Have make specify build-time constants for file headers
#include "buildInfo.h"

constexpr unsigned int   N_TRIALS = 3;

constexpr unsigned int   N_THREAD_TEAMS = 1;
constexpr unsigned int   N_THREADS_PER_TEAM = 7;
constexpr int            N_STREAMS = 32; 
constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 12884901888; 

// Assuming Summit with 7 cores/MPI process (i.e. one GPU/process)
// and no hardware threading
constexpr unsigned int   N_THREADS_PER_PROC = 7;
constexpr unsigned int   N_BLOCKS = N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z;
const     unsigned int   MAX_BLOCKS_PER_PACKET = std::min(N_BLOCKS,
                                                          static_cast<unsigned int>(10));

void  setUp(void) { }

void  tearDown(const std::string& filename,
               const std::string& mode,
               const int nHostThreads, 
               const int nGpuThreads, 
               const int nBlocksPerPacket,
               const double walltime) {
    using namespace orchestration;

    int   rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::ofstream  fptr;
        fptr.open(filename, std::ios::out | std::ios::app);
        fptr << std::setprecision(15) 
             << mode << ","
             << (nHostThreads < 0 ? "NaN" : std::to_string(nHostThreads)) << ","
             << (nGpuThreads  < 0 ? "NaN" : std::to_string(nGpuThreads)) << ","
             << NXB << "," << NYB << "," << NZB << ","
             << N_BLOCKS_X << ","
             << N_BLOCKS_Y << ","
             << N_BLOCKS_Z << ","
             << (nBlocksPerPacket < 1 ? "NaN" : std::to_string(nBlocksPerPacket)) << ","
             << walltime << std::endl;
        fptr.close();
    }
}

int   main(int argc, char* argv[]) {
    using namespace orchestration;

    // Initialize simulation
    Logger::instantiate("GatherDataCudaCpp.log");

    Runtime::instantiate(N_THREAD_TEAMS, N_THREADS_PER_TEAM,
                         N_STREAMS, MEMORY_POOL_SIZE_BYTES);
    Runtime&   runtime = orchestration::Runtime::instance();

    Grid::instantiate();
    Grid&   grid = Grid::instance();
    grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu);

    // Setup logging of results
    std::string  fname("gatherDataCudaCpp_");
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
    fptr << "pmode,n_thd_cpu,n_thd_gpu,"
         << "NXB,NYB,NZB,N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z,"
         << "n_blks_per_packet,"
         << "Walltime_sec\n";
    fptr.close();

    std::string   testName  = "NXB=" + std::to_string(NXB) + " / ";
                  testName += "NYB=" + std::to_string(NYB) + " / ";
                  testName += "NZB=" + std::to_string(NZB) + " / ";
                  testName += "N Blocks X=" + std::to_string(N_BLOCKS_X) + " / ";
                  testName += "N Blocks Y=" + std::to_string(N_BLOCKS_Y) + " / ";
                  testName += "N Blocks Z=" + std::to_string(N_BLOCKS_Z);
    Logger::instance().log("[Simulation] Start " + testName);

    RuntimeAction    sleepyGoByeBye;
    sleepyGoByeBye.name = "sleepyGoByeBye";
    for (unsigned int j=0; j<N_TRIALS; ++j) {
        Logger::instance().log("[Simulation] Start trial " + std::to_string(j+1));

        /***** SERIAL TEST - Three iteration loops  *****/
        Logger::instance().log("[Simulation] Start Serial Test");

        setUp();

        double tStart = MPI_Wtime(); 
        std::unique_ptr<Tile>    dataItem{};
        for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next()) {
            dataItem = ti->buildCurrentTile();
            ActionRoutines::sleepyGoByeBye_tile_cpu(0, dataItem.get());
        }
        double tWalltime = MPI_Wtime() - tStart;
        tearDown(fname, "Serial", -1, -1, -1, tWalltime);
        Logger::instance().log("[Simulation] End Serial Test");

        /***** RUNTIME TEST - Multithreaded Runtime/Host version *****/
        std::string    msg;
        Logger::instance().log("[Simulation] Start Host-Only Runtime Test");
        sleepyGoByeBye.teamType        = ThreadTeamDataType::BLOCK;
        sleepyGoByeBye.nTilesPerPacket = 0;
        sleepyGoByeBye.routine         = ActionRoutines::sleepyGoByeBye_tile_cpu;
        for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-1; ++nThreads) {
            sleepyGoByeBye.nInitialThreads = nThreads;

            setUp();

            tStart = MPI_Wtime(); 
            runtime.executeCpuTasks("SleepyTime", sleepyGoByeBye);
            tWalltime = MPI_Wtime() - tStart; 

            // We have one host thread acting as action parallel distributor
            tearDown(fname, "Runtime", nThreads+1, 0, -1, tWalltime);
        }
        Logger::instance().log("[Simulation] End Host-Only Runtime Test");

#if defined(USE_CUDA_BACKEND)
        /***** RUNTIME TEST - GPU version *****/
        Logger::instance().log("[Simulation] Start GPU-Only Runtime Test");
        sleepyGoByeBye.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
        sleepyGoByeBye.routine  = ActionRoutines::sleepyGoByeBye_packet_cuda_gpu;
        for (unsigned int nBlks=1; nBlks<=MAX_BLOCKS_PER_PACKET; ++nBlks) {
            for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-1; ++nThreads) {
                sleepyGoByeBye.nInitialThreads = nThreads;
                sleepyGoByeBye.nTilesPerPacket = nBlks;

                setUp();

                tStart = MPI_Wtime(); 
                runtime.executeGpuTasks("SleepyTime", sleepyGoByeBye);
                tWalltime = MPI_Wtime() - tStart; 

                // We have one host thread acting as action parallel distributor
                tearDown(fname, "Runtime", 1, nThreads, nBlks, tWalltime);
            }
        }
        Logger::instance().log("[Simulation] End GPU-Only Runtime Test");
#endif

        Logger::instance().log("[Simulation] End trial " + std::to_string(j+1));
    }

    grid.destroyDomain();

    Logger::instance().log("[Simulation] End " + testName);

    return 0;
}

