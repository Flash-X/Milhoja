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
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"
#include "computeLaplacianFused.h"
#include "Analysis.h"

#ifdef USE_CUDA_BACKEND
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"
#endif

// Have make specify build-time constants for file headers
#include "buildInfo.h"

// No AMR.  Set level to 0-based AMReX coarse level
constexpr unsigned int   LEVEL = 0;


constexpr unsigned int   N_TRIALS = 50;

constexpr unsigned int   N_THREAD_TEAMS = 2;
constexpr unsigned int   MAX_THREADS = 7;
constexpr int            N_STREAMS = 32; 
constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 12884901888; 

// Assuming Summit with 7 cores/MPI process (i.e. one GPU/process)
// and no hardware threading
constexpr unsigned int   N_THREADS_PER_PROC = 7;

// Allow for the largest packet to contain all blocks
constexpr unsigned int   N_BLOCKS = N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z;
constexpr unsigned int   PACKET_STEP_SIZE = 10;

constexpr unsigned int   TURN_STEP_SIZE = 5;
constexpr unsigned int   MAX_TURN_SIZE  = 60;

void  setUp(void) {
    using namespace orchestration;

    RuntimeAction    setICs;
    setICs.name            = "SetICs";
    setICs.nInitialThreads = N_THREADS_PER_PROC - 1;
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
               const unsigned int nTilesPerPacket,
               const unsigned int nTilesPerCpuTurn,
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
        computeError.nInitialThreads = N_THREADS_PER_PROC - 1;
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
             << (nTilesPerPacket < 1 ? "NaN" : std::to_string(nTilesPerPacket)) << ","
             << (nTilesPerCpuTurn < 1 ? "NaN" : std::to_string(nTilesPerCpuTurn)) << ","
             << NXB << "," << NYB << "," << NZB << ","
             << N_BLOCKS_X << ","
             << N_BLOCKS_Y << ","
             << N_BLOCKS_Z << ","
             << dx << "," << dy << ","
             << L_inf_dens << "," << L_inf_ener << ","
             << walltime << std::endl;
        fptr.close();
    }

#ifdef USE_CUDA_BACKEND
    orchestration::CudaMemoryManager::instance().reset();
#endif
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
    fptr << "pmode,n_loops,n_thd_host,n_thd_gpu,n_tiles_per_packet,n_tiles_per_cpu_turn,"
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

        /***** SERIAL TEST - Single iteration loop  *****/
        Logger::instance().log("[Simulation] Start Single Loop Serial Test");

        setUp();

        double   tStart = MPI_Wtime(); 
        std::unique_ptr<Tile>    dataItem{};
        for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next()) {
            dataItem = ti->buildCurrentTile();
            ActionRoutines::computeLaplacianDensity_tile_cpu(0, dataItem.get());
            ActionRoutines::computeLaplacianEnergy_tile_cpu(0, dataItem.get());
        }
        double   tWalltime = MPI_Wtime() - tStart; 

        tearDown(fname, "Serial", 1, 1, 0, 0, 0, tWalltime);
        Logger::instance().log("[Simulation] End Single Loop Serial Test");

        /***** RUNTIME TEST - Multithreaded Serial execution on host *****/
        Logger::instance().log("[Simulation] Start 2 Loop Serialized Runtime Test");
        RuntimeAction    computeLaplacianDensity;
        RuntimeAction    computeLaplacianEnergy;

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

        for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-1; ++nThreads) {
            computeLaplacianDensity.nInitialThreads = nThreads;
            computeLaplacianEnergy.nInitialThreads  = nThreads;

            setUp();

            tStart = MPI_Wtime(); 
            runtime.executeCpuTasks("LapDens", computeLaplacianDensity);
            runtime.executeCpuTasks("LapEner", computeLaplacianEnergy);
            tWalltime = MPI_Wtime() - tStart; 

            tearDown(fname, "Runtime", 2, nThreads+1, 0, 0, 0, tWalltime);
        }
        Logger::instance().log("[Simulation] End 2 Loop Serialized Runtime Test");

        /***** RUNTIME TEST - Fused, multithreaded, serial execution on host *****/
        Logger::instance().log("[Simulation] Start 2 Loop Serialized Runtime Test - Fused");

        RuntimeAction    computeLaplacianFused_cpu;

        computeLaplacianFused_cpu.name            = "LaplacianFused_cpu";
        computeLaplacianFused_cpu.nInitialThreads = 1;
        computeLaplacianFused_cpu.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianFused_cpu.nTilesPerPacket = 0;
        computeLaplacianFused_cpu.routine         = ActionRoutines::computeLaplacianFusedKernels_tile_cpu;

        for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-1; ++nThreads) {
            computeLaplacianFused_cpu.nInitialThreads = nThreads;

            setUp();

            tStart = MPI_Wtime(); 
            runtime.executeCpuTasks("LapFused", computeLaplacianFused_cpu);
            tWalltime = MPI_Wtime() - tStart; 

            tearDown(fname, "RuntimeFused", 1, nThreads+1, 0, 0, 0, tWalltime);
        }
        Logger::instance().log("[Simulation] End 2 Loop Serialized Runtime Test - Fused");

#if defined(USE_CUDA_BACKEND)
        /***** RUNTIME TEST - GPU-only *****/
        Logger::instance().log("[Simulation] Start GPU-Only Runtime Test");

        computeLaplacianDensity.name            = "LaplacianDensity";
        computeLaplacianDensity.nInitialThreads = 3;
        computeLaplacianDensity.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianDensity.nTilesPerPacket = 20;
        computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

        computeLaplacianEnergy.name            = "LaplacianEnergy";
        computeLaplacianEnergy.nInitialThreads = 3;
        computeLaplacianEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianEnergy.nTilesPerPacket = 20;
        computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

        unsigned int   nBlks = 0;
        while (nBlks < N_BLOCKS) {
            nBlks = std::min(N_BLOCKS, nBlks + PACKET_STEP_SIZE);
            computeLaplacianDensity.nTilesPerPacket = nBlks;
            computeLaplacianEnergy.nTilesPerPacket  = nBlks;

            for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-1; ++nThreads) {
                computeLaplacianDensity.nInitialThreads = nThreads;
                computeLaplacianEnergy.nInitialThreads  = nThreads;

                setUp();

                tStart = MPI_Wtime(); 
                runtime.executeGpuTasks("LapDens", computeLaplacianDensity);
                runtime.executeGpuTasks("LapEner", computeLaplacianEnergy);
                tWalltime = MPI_Wtime() - tStart; 

                tearDown(fname, "Runtime", 2, 1, nThreads, nBlks, 0, tWalltime);
            }
        }
        Logger::instance().log("[Simulation] End GPU-Only Runtime Test");

        /***** RUNTIME TEST - CPU/GPU *****/
        Logger::instance().log("[Simulation] Start CPU/GPU Runtime Test");

        computeLaplacianDensity.name            = "LaplacianDensity_cpu";
        computeLaplacianDensity.nInitialThreads = 3;
        computeLaplacianDensity.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianDensity.nTilesPerPacket = 0;
        computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

        computeLaplacianEnergy.name            = "LaplacianEnergy_gpu";
        computeLaplacianEnergy.nInitialThreads = 3;
        computeLaplacianEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianEnergy.nTilesPerPacket = 20;
        computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

        nBlks = 0;
        while (nBlks < N_BLOCKS) {
            nBlks = std::min(N_BLOCKS, nBlks + PACKET_STEP_SIZE);
            computeLaplacianEnergy.nTilesPerPacket = nBlks;

            for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-2; ++nThreads) {
                unsigned int nThdHost = N_THREADS_PER_PROC - nThreads - 1;
                unsigned int nThdGpu  = nThreads;

                computeLaplacianDensity.nInitialThreads = nThdHost;
                computeLaplacianEnergy.nInitialThreads  = nThdGpu;

                setUp();

                tStart = MPI_Wtime(); 
                runtime.executeCpuGpuTasks("ConcurrentCpuGpu",
                                           computeLaplacianDensity,
                                           computeLaplacianEnergy);
                tWalltime = MPI_Wtime() - tStart; 

                tearDown(fname, "Runtime", 1, nThdHost+1, nThdGpu, nBlks, 0, tWalltime);
            }
        }
        Logger::instance().log("[Simulation] End CPU/GPU Runtime Test");

        /***** RUNTIME TEST - GPU-only Fused *****/
        Logger::instance().log("[Simulation] Start GPU-Only Runtime Test - Fused");

        RuntimeAction    computeLaplacianFused_gpu;

        computeLaplacianFused_gpu.name            = "LaplacianFused_gpu";
        computeLaplacianFused_gpu.nInitialThreads = 3;
        computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianFused_gpu.nTilesPerPacket = 20;
        computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsStrong_packet_oacc_summit;

        nBlks = 0;
        while (nBlks < N_BLOCKS) {
            nBlks = std::min(N_BLOCKS, nBlks + PACKET_STEP_SIZE);
            computeLaplacianFused_gpu.nTilesPerPacket = nBlks;

            for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-1; ++nThreads) {
                computeLaplacianFused_gpu.nInitialThreads = nThreads;

                setUp();

                tStart = MPI_Wtime(); 
                runtime.executeGpuTasks("LapFused", computeLaplacianFused_gpu);
                tWalltime = MPI_Wtime() - tStart; 

                tearDown(fname, "RuntimeFused", 1, 1, nThreads, nBlks, 0, tWalltime);
            }
        }
        Logger::instance().log("[Simulation] End GPU-Only Runtime Test - Fused");

        /***** RUNTIME TEST - CPU/GPU Shared Fused *****/
        Logger::instance().log("[Simulation] Start Data Parallel Cpu/Gpu - Fused");

        computeLaplacianFused_cpu.name            = "LaplacianFused_cpu";
        computeLaplacianFused_cpu.nInitialThreads = 3;
        computeLaplacianFused_cpu.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianFused_cpu.nTilesPerPacket = 0;
        computeLaplacianFused_cpu.routine         = ActionRoutines::computeLaplacianFusedKernels_tile_cpu;

        computeLaplacianFused_gpu.name            = "LaplacianFused_gpu";
        computeLaplacianFused_gpu.nInitialThreads = 3;
        computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianFused_gpu.nTilesPerPacket = 20;
        computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsStrong_packet_oacc_summit;

        // Don't allow the CPU to perform all work
        unsigned int maxTurnSize = 0;
        if (N_BLOCKS < 2*TURN_STEP_SIZE) {
            // If the number of blocks is too small compared to the desired
            // step size, then shrink step and prefer to give more work to CPU
            // Note that maxTurnSize < TURN_STEP_SIZE
            maxTurnSize = ceil(N_BLOCKS / 2.0);
        } else {
            maxTurnSize = std::min(N_BLOCKS - TURN_STEP_SIZE, MAX_TURN_SIZE);
        }

        // TODO: I think that this is gathering too much data.  For instance, if
        // the number of blocks is low and the number of blocks to be done in the
        // first CPU turn is just less than this, then the only data packet will
        // have just a few blocks.  There is no sense in repeating the data
        // gathering for ever larger blocks that will never be completely
        // filled.
        nBlks = 0;
        while (nBlks < N_BLOCKS) {
            nBlks = std::min(N_BLOCKS, nBlks + PACKET_STEP_SIZE);
            computeLaplacianFused_gpu.nTilesPerPacket = nBlks;

            for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-2; ++nThreads) {
                unsigned int nThdHost = N_THREADS_PER_PROC - nThreads - 1;
                unsigned int nThdGpu  = nThreads;
                computeLaplacianFused_cpu.nInitialThreads = nThdHost;
                computeLaplacianFused_gpu.nInitialThreads = nThdGpu;

                unsigned int    tilesPerTurn = 0;
                while (tilesPerTurn < maxTurnSize) {
                    tilesPerTurn = std::min(maxTurnSize, tilesPerTurn + TURN_STEP_SIZE);

                    setUp();

                    tStart = MPI_Wtime(); 
                    runtime.executeCpuGpuSplitTasks("DataParallelFused",
                                                    computeLaplacianFused_cpu,
                                                    computeLaplacianFused_gpu,
                                                    tilesPerTurn);
                    tWalltime = MPI_Wtime() - tStart; 

                    tearDown(fname, "RuntimeFused", 1, nThdHost+1, nThdGpu, nBlks, tilesPerTurn, tWalltime);
                }
            }
        }
        Logger::instance().log("[Simulation] End Data Parallel Cpu/Gpu - Fused");

        /***** RUNTIME TEST - GPU-only Fused Actions *****/
        Logger::instance().log("[Simulation] Start GPU-Only Runtime Test - Fused Actions");

        computeLaplacianFused_gpu.name            = "LaplacianFused_gpu";
        computeLaplacianFused_gpu.nInitialThreads = 3;
        computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianFused_gpu.nTilesPerPacket = 20;
        computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit;

        nBlks = 0;
        while (nBlks < N_BLOCKS) {
            nBlks = std::min(N_BLOCKS, nBlks + PACKET_STEP_SIZE);
            computeLaplacianFused_gpu.nTilesPerPacket = nBlks;

            for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-1; ++nThreads) {
                computeLaplacianFused_gpu.nInitialThreads = nThreads;

                setUp();

                tStart = MPI_Wtime(); 
                runtime.executeGpuTasks("LapFused", computeLaplacianFused_gpu);
                tWalltime = MPI_Wtime() - tStart; 

                tearDown(fname, "RuntimeFusedActions", 1, 1, nThreads, nBlks, 0, tWalltime);
            }
        }
        Logger::instance().log("[Simulation] End GPU-Only Runtime Test - Fused Actions");

        /***** RUNTIME TEST - CPU/GPU Shared Fused *****/
        Logger::instance().log("[Simulation] Start Data Parallel Cpu/Gpu - Fused Actions");

        computeLaplacianFused_cpu.name            = "LaplacianFused_cpu";
        computeLaplacianFused_cpu.nInitialThreads = 3;
        computeLaplacianFused_cpu.teamType        = ThreadTeamDataType::BLOCK;
        computeLaplacianFused_cpu.nTilesPerPacket = 0;
        computeLaplacianFused_cpu.routine         = ActionRoutines::computeLaplacianFusedKernels_tile_cpu;

        computeLaplacianFused_gpu.name            = "LaplacianFused_gpu";
        computeLaplacianFused_gpu.nInitialThreads = 3;
        computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianFused_gpu.nTilesPerPacket = 20;
        computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit;

        // Don't allow the CPU to perform all work
        maxTurnSize = 0;
        if (N_BLOCKS < 2*TURN_STEP_SIZE) {
            // If the number of blocks is too small compared to the desired
            // step size, then shrink step and prefer to give more work to CPU
            // Note that maxTurnSize < TURN_STEP_SIZE
            maxTurnSize = ceil(N_BLOCKS / 2.0);
        } else {
            maxTurnSize = std::min(N_BLOCKS - TURN_STEP_SIZE, MAX_TURN_SIZE);
        }

        // TODO: I think that this is gathering too much data.  For instance, if
        // the number of blocks is low and the number of blocks to be done in the
        // first CPU turn is just less than this, then the only data packet will
        // have just a few blocks.  There is no sense in repeating the data
        // gathering for ever larger blocks that will never be completely
        // filled.
        nBlks = 0;
        while (nBlks < N_BLOCKS) {
            nBlks = std::min(N_BLOCKS, nBlks + PACKET_STEP_SIZE);
            computeLaplacianFused_gpu.nTilesPerPacket = nBlks;

            for (unsigned int nThreads=1; nThreads<=N_THREADS_PER_PROC-2; ++nThreads) {
                unsigned int nThdHost = N_THREADS_PER_PROC - nThreads - 1;
                unsigned int nThdGpu  = nThreads;
                computeLaplacianFused_cpu.nInitialThreads = nThdHost;
                computeLaplacianFused_gpu.nInitialThreads = nThdGpu;

                unsigned int    tilesPerTurn = 0;
                while (tilesPerTurn < maxTurnSize) {
                    tilesPerTurn = std::min(maxTurnSize, tilesPerTurn + TURN_STEP_SIZE);

                    setUp();

                    tStart = MPI_Wtime(); 
                    runtime.executeCpuGpuSplitTasks("DataParallelFused",
                                                    computeLaplacianFused_cpu,
                                                    computeLaplacianFused_gpu,
                                                    tilesPerTurn);
                    tWalltime = MPI_Wtime() - tStart; 

                    tearDown(fname, "RuntimeFusedActions", 1, nThdHost+1, nThdGpu, nBlks, tilesPerTurn, tWalltime);
                }
            }
        }
        Logger::instance().log("[Simulation] End Data Parallel Cpu/Gpu - Fused Actions");
#endif

        Logger::instance().log("[Simulation] End trial " + std::to_string(j+1));
    }

    grid.destroyDomain();

    Logger::instance().log("[Simulation] End " + testName);

    return 0;
}

