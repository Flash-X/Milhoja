#include "Milhoja_Runtime.h"

#include <chrono>
#include <thread>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#ifdef USE_THREADED_DISTRIBUTOR
#include <omp.h>
#endif

#include "Milhoja.h"
#ifdef RUNTIME_CAN_USE_TILEITER
#include "Milhoja_Grid.h"
#endif
#ifndef RUNTIME_MUST_USE_TILEITER
#include "Milhoja_FlashxrTileRaw.h"
#include "Milhoja_TileFlashxr.h"
#endif
#include "Milhoja_RuntimeBackend.h"
#include "Milhoja_DataPacket.h"
#include "Milhoja_Logger.h"

namespace milhoja {

unsigned int    Runtime::nTeams_            = 0;
unsigned int    Runtime::maxThreadsPerTeam_ = 0;
bool            Runtime::initialized_       = false;
bool            Runtime::finalized_         = false;

/**
 * 
 *
 */
void   Runtime::initialize(const unsigned int nTeams,
                            const unsigned int nThreadsPerTeam,
                            const unsigned int nStreams,
                            const std::size_t  nBytesInCpuMemoryPool,
                            const std::size_t  nBytesInGpuMemoryPools) {
    // finalized_ => initialized_
    // Therefore, no need to check finalized_.
    if (initialized_) {
        throw std::logic_error("[Runtime::initialize] Already initialized");
    } else if (nTeams == 0) {
        throw std::invalid_argument("[Runtime::initialize] "
                                    "Need at least one ThreadTeam");
    } else if (nThreadsPerTeam == 0) {
        throw std::invalid_argument("[Runtime::initialize] "
                                    "Need at least one thread per team");
    }

    Logger::instance().log("[Runtime] Initializing...");

    nTeams_ = nTeams;
    maxThreadsPerTeam_ = nThreadsPerTeam;
    initialized_ = true;

    milhoja::RuntimeBackend::initialize(nStreams,
                                        nBytesInCpuMemoryPool,
                                        nBytesInGpuMemoryPools);

    // Create/initialize runtime
    instance();

    Logger::instance().log("[Runtime] Created and ready for use");
}

/**
 *
 */
void   Runtime::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[Runtime::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[Runtime::finalize] Already finalized");
    }

    Logger::instance().log("[Runtime] Finalizing ...");

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }
    delete [] teams_;
    teams_ = nullptr;

#ifndef RUNTIME_MUST_USE_TILEITER
    packet_gpu_.reset();
#endif

    milhoja::RuntimeBackend::instance().finalize();

    finalized_ = true;

    Logger::instance().log("[Runtime] Finalized");
#if !(defined(RUNTIME_CAN_USE_TILEITER) || defined(FULL_MILHOJAGRID))
    Logger::instance().finalize();
#endif
}

/**
 * 
 *
 * \return 
 */
Runtime& Runtime::instance(void) {
    if (!initialized_) {
        throw std::logic_error("[Runtime::instance] Initialize first");
    } else if (finalized_) {
        throw std::logic_error("[Runtime::instance] No access after finalization");
    }

    static Runtime     singleton;
    return singleton;
}

/**
 * 
 *
 */
Runtime::Runtime(void)
    : teams_{nullptr}
#ifndef RUNTIME_MUST_USE_TILEITER
    ,packet_gpu_{}
#endif
{
    teams_ = new ThreadTeam*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam(maxThreadsPerTeam_, i);
    }
}

/**
 *
 */
 Runtime::~Runtime(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[Runtime::~Runtime] ERROR - Not finalized" << std::endl;
    }
 }

/**
 * 
 *
 */
#ifndef RUNTIME_MUST_USE_TILEITER
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
void Runtime::setupPipelineForCpuTasks(const std::string& actionName,
                              const RuntimeAction& cpuAction) {
    Logger::instance().log("[Runtime] Start setting up single CPU action");

    if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime:setupPipelineForCpuTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime:setupPipelineForCpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime:setupPipelineForCpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    ThreadTeam*   cpuTeam = teams_[0];

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads = cpuAction.nInitialThreads + 1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime:setupPipelineForCpuTasks] "
                               "CPU team could receive too many thread "
                               "activation calls from distributor");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "CPU_Block_Team");

    Logger::instance().log("[Runtime] End setting up CPU action");
}

void Runtime::pushTileToPipeline(const std::string& actionName,
                                 const TileWrapper& prototype,
                                 const FlashxrTileRawPtrs& tP,
                                 const FlashxTileRawInts& tI,
                                 const FlashxTileRawReals& tR
                                 ) {
#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Push single tile task to single CPU pipeline");
#endif
    ThreadTeam*   cpuTeam = teams_[0];

    //    cpuTeam->enqueue( prototype.clone( std::unique_ptr<Tile>{new TileFlashxr{tP, tI, tR}} ) );
    cpuTeam->enqueue( prototype.clone( std::make_unique<TileFlashxr>(tP, tI, tR) ) );

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Single tile task was pushed to CPU pipeline");
#endif
}
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
void Runtime::teardownPipelineForCpuTasks(const std::string& actionName) {
    ThreadTeam*   cpuTeam = teams_[0];
    cpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread 
    // in team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();

    // No need to break apart the thread team configuration

    Logger::instance().log("[Runtime] End single CPU action");
}
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// void Runtime::executeCpuTasks(const std::string& actionName,
//                               const RuntimeAction& cpuAction,
//                               const TileWrapper& prototype) {
//     Logger::instance().log("[Runtime] Putting it all together...");
//     setupPipelineForCpuTasks(actionName,
//                              cpuAction,
//                              prototype);
//     runPipelineForCpuTasks(actionName,
//                            cpuAction,
//                            prototype);
//     teardownPipelineForCpuTasks(actionName,
//                                 cpuAction,
//                                 prototype);
// }
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#endif
#ifdef RUNTIME_CAN_USE_TILEITER
  // Traditional execute-style runtime invocation
void Runtime::executeCpuTasks(const std::string& actionName,
                              const RuntimeAction& cpuAction,
                              const TileWrapper& prototype) {
    Logger::instance().log("[Runtime] Start single CPU action");

    if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeCpuTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime::executeCpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    ThreadTeam*   cpuTeam = teams_[0];

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads = cpuAction.nInitialThreads + 1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuTasks] "
                               "CPU team could receive too many thread "
                               "activation calls from distributor");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "CPU_Block_Team");

    //***** ACTION PARALLEL DISTRIBUTOR
    Grid&   grid = Grid::instance();
    for (unsigned int level=0; level<=grid.getMaxLevel(); ++level) {
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            cpuTeam->enqueue( prototype.clone( ti->buildCurrentTile() ) );
        }
    }
    cpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread 
    // in team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();

    // No need to break apart the thread team configuration

    Logger::instance().log("[Runtime] End single CPU action");
}
#endif
/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeGpuTasks(const std::string& bundleName,
                              const unsigned int nDistributorThreads,
                              const unsigned int stagger_usec,
                              const RuntimeAction& gpuAction,
                              const DataPacket& packetPrototype) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif

    Logger::instance().log("[Runtime] Start single GPU action");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads / stagger of " 
                        + std::to_string(stagger_usec)
                        + " us";
    Logger::instance().log(msg);

    if        (nDistThreads <= 0) {
        throw std::invalid_argument("[Runtime::executedGpuTasks] "
                                    "nDistributorThreads must be positive");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeGpuTasks] "
                               "Given GPU action should run on "
                               "data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeGpuTasks] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime::executeGpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       gpuTeam   = teams_[0];
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    //***** START EXECUTION CYCLE
    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                  level = 0;
    Grid&                         grid = Grid::instance();
    RuntimeBackend&               backend = RuntimeBackend::instance();
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, backend, level, \
                     packetPrototype, stagger_usec, \
                     gpuTeam, gpuAction) \
                     num_threads(nDistThreads)
#endif
    {
#ifdef USE_THREADED_DISTRIBUTOR
        int         tIdx = omp_get_thread_num();
#else
        int         tIdx = 0;
#endif

        std::this_thread::sleep_for(std::chrono::microseconds(tIdx * stagger_usec));

        std::shared_ptr<DataPacket>   packet_gpu = packetPrototype.clone();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            packet_gpu->addTile( ti->buildCurrentTile() );
            if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
                packet_gpu->pack();
                backend.initiateHostToGpuTransfer(*(packet_gpu.get()));

                gpuTeam->enqueue( std::move(packet_gpu) );

                packet_gpu = packetPrototype.clone();
            }
        }

        if (packet_gpu->nTiles() > 0) {
            packet_gpu->pack();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            gpuTeam->enqueue( std::move(packet_gpu) );
        } else {
            packet_gpu.reset();
        }
    } // implied barrier
    gpuTeam->closeQueue(nullptr);
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime] End single GPU action");
}
#  endif  // #ifdef RUNTIME_CAN_USE_TILEITER

#  ifndef RUNTIME_MUST_USE_TILEITER
void Runtime::setupPipelineForGpuTasks(const std::string& bundleName,
                              const unsigned int stagger_usec,
                              const RuntimeAction& gpuAction,
                              const DataPacket& packetPrototype) {

    Logger::instance().log("[Runtime] Start setting up single GPU action");

    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime:setupPipelineForGpuTasks] "
                               "Given GPU action should run on "
                               "data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime:setupPipelineForGpuTasks] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime:setupPipelineForGpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }
    nTilesPerPacket_ = gpuAction.nTilesPerPacket;

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       gpuTeam   = teams_[0];
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    //***** START EXECUTION CYCLE
    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");
    gpuToHost1_.startCycle();

    {
#ifdef USE_THREADED_DISTRIBUTOR
        int         tIdx = omp_get_thread_num();
        std::this_thread::sleep_for(std::chrono::microseconds(tIdx * stagger_usec));
#endif

        packet_gpu_ = packetPrototype.clone();
    }

    Logger::instance().log("[Runtime] End setting up single GPU action");
}
void Runtime::pushTileToGpuPipeline(const std::string& bundleName,
                                    const DataPacket& packetPrototype,
                                    const FlashxrTileRawPtrs& tP,
                                    const FlashxTileRawInts& tI,
                                    const FlashxTileRawReals& tR) {

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Push single tile task to single GPU pipeline");
#endif

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:pushTileToGpuPipeline] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime:pushTileToGpuPipeline] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLED THREAD TEAM CONFIGURATION
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       gpuTeam   = teams_[0];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    {
      //            packet_gpu_->addTile( std::unique_ptr<Tile>{new TileFlashxr{tP, tI, tR}} );
                  packet_gpu_->addTile( static_cast<std::shared_ptr<Tile> >(std::make_unique<TileFlashxr>(tP, tI, tR) ));
                  //      packet_gpu_->addTile( std::make_shared<TileFlashxr>(tP, tI, tR) );
            if (packet_gpu_->nTiles() >= nTilesPerPacket_) {
                packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
                Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " tiles...");
#endif
                backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));

                gpuTeam->enqueue( std::move(packet_gpu_) );

                packet_gpu_ = packetPrototype.clone();
            }
    }

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Single tile task was pushed to GPU pipeline");
#endif
}
void Runtime::teardownPipelineForGpuTasks(const std::string& bundleName) {

    Logger::instance().log("[Runtime] Tear Down single GPU action");

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:teardownPipelineForGpuTasks] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime:teardownPipelineForGpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLED THREAD TEAM CONFIGURATION
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       gpuTeam   = teams_[0];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    {

        if (packet_gpu_->nTiles() > 0) {
            packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
            Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " final tiles...");
#endif
            backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));
            gpuTeam->enqueue( std::move(packet_gpu_) );
        } else {
            packet_gpu_.reset();
        }
    }
    gpuTeam->closeQueue(nullptr);
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime:teardownPipelineForGpuTasks] End single GPU action");
}

#  endif

#endif   // #ifdef RUNTIME_SUPPORT_DATAPACKETS
/**
 * 
 *
 * \return 
 */

#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef MILHOJA_TIMED_PIPELINE_CONFIGS
#    ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeGpuTasks_timed(const std::string& bundleName,
                                    const unsigned int nDistributorThreads,
                                    const unsigned int stagger_usec,
                                    const RuntimeAction& gpuAction,
                                    const DataPacket& packetPrototype,
                                    const unsigned int stepNumber,
                                    const MPI_Comm comm) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif

    Logger::instance().log("[Runtime] Start single GPU action (Timed)");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads / stagger of " 
                        + std::to_string(stagger_usec)
                        + " us";
    Logger::instance().log(msg);

    if        (nDistThreads <= 0) {
        throw std::invalid_argument("[Runtime::executedGpuTasks_timed] "
                                    "nDistributorThreads must be positive");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeGpuTasks_timed] "
                               "Given GPU action should run on "
                               "data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeGpuTasks_timed] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime::executeGpuTasks_timed] "
                               "Need at least one ThreadTeam in runtime");
    }

    Grid&                grid = Grid::instance();
    RuntimeBackend&      backend = RuntimeBackend::instance();

    //***** SETUP TIMING
    int rank = -1;
    MPI_Comm_rank(comm, &rank);

    unsigned int   nxb = 1;
    unsigned int   nyb = 1;
    unsigned int   nzb = 1;
    grid.getBlockSize(&nxb, &nyb, &nzb);

    unsigned int  nPackets = ceil(  (double)grid.getNumberLocalBlocks()
                                  / (double)gpuAction.nTilesPerPacket);

    unsigned int  pCounts[nDistThreads];
    unsigned int  bCounts[nDistThreads][nPackets];
    double        wtimesPack_sec[nDistThreads][nPackets];
    double        wtimesAsync_sec[nDistThreads][nPackets];
    double        wtimesPacket_sec[nDistThreads][nPackets];

    std::string   filename("timings_packet_step");
    filename += std::to_string(stepNumber);
    filename += "_rank";
    filename += std::to_string(rank);
    filename += ".dat";

    std::ofstream   fptr;
    fptr.open(filename, std::ios::out);
    fptr << "# Testname = GPU-Only\n";
    fptr << "# Step = " << stepNumber << "\n";
    fptr << "# MPI rank = " << rank << "\n";
    fptr << "# Dimension = " << MILHOJA_NDIM << "\n";
    fptr << "# NXB = " << nxb << "\n";
    fptr << "# NYB = " << nyb << "\n";
    fptr << "# NZB = " << nzb << "\n";
    fptr << "# n_distributor_threads = " << nDistThreads << "\n";
    fptr << "# stagger_usec = " << stagger_usec << "\n";
    fptr << "# n_cpu_threads = 0\n";
    fptr << "# n_gpu_threads = " << gpuAction.nInitialThreads << "\n";
    fptr << "# n_blocks_per_packet = " << gpuAction.nTilesPerPacket << "\n";
    fptr << "# n_blocks_per_cpu_turn = 0\n";
    fptr << "# MPI_Wtick_sec = " << MPI_Wtick() << "\n";
    fptr << "# thread,packet,nblocks,walltime_pack_sec,walltime_async_sec,walltime_packet_sec\n";

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       gpuTeam   = teams_[0];
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    //***** START EXECUTION CYCLE
    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int  level = 0;
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, backend, level, packetPrototype, \
                            gpuTeam, gpuAction, stagger_usec, \
                            wtimesPack_sec, wtimesAsync_sec, wtimesPacket_sec, \
                            pCounts, bCounts) \
                     num_threads(nDistThreads)
#endif
    {
#ifdef USE_THREADED_DISTRIBUTOR
        int         tIdx = omp_get_thread_num();
#else
        int         tIdx = 0;
#endif
        unsigned int  pIdx         = 0;
        double        tStartPacket = 0.0;
        double        tStartPack   = 0.0;
        double        tStartAsync  = 0.0;

        std::this_thread::sleep_for(std::chrono::microseconds(tIdx * stagger_usec));

        tStartPacket = MPI_Wtime();
        std::shared_ptr<DataPacket>   packet_gpu = packetPrototype.clone();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            packet_gpu->addTile( ti->buildCurrentTile() );
            if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
                tStartPack = MPI_Wtime();
                packet_gpu->pack();
                wtimesPack_sec[tIdx][pIdx] = MPI_Wtime() - tStartPack;

                tStartAsync = MPI_Wtime();
                backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
                wtimesAsync_sec[tIdx][pIdx] = MPI_Wtime() - tStartAsync;

                bCounts[tIdx][pIdx] = packet_gpu->nTiles();
                gpuTeam->enqueue( std::move(packet_gpu) );

                wtimesPacket_sec[tIdx][pIdx] = MPI_Wtime() - tStartPacket;
                ++pIdx;

                tStartPacket = MPI_Wtime();
                packet_gpu = packetPrototype.clone();
            }
        }

        if (packet_gpu->nTiles() > 0) {
            tStartPack = MPI_Wtime();
            packet_gpu->pack();
            wtimesPack_sec[tIdx][pIdx] = MPI_Wtime() - tStartPack;

            tStartAsync = MPI_Wtime();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            wtimesAsync_sec[tIdx][pIdx] = MPI_Wtime() - tStartAsync;

            bCounts[tIdx][pIdx] = packet_gpu->nTiles();
            gpuTeam->enqueue( std::move(packet_gpu) );

            wtimesPacket_sec[tIdx][pIdx] = MPI_Wtime() - tStartPacket;

            ++pIdx;
        } else {
            packet_gpu.reset();
        }

        pCounts[tIdx] = pIdx;
    } // implied barrier
    gpuTeam->closeQueue(nullptr);

    fptr << std::setprecision(15);
    for     (unsigned int tIdx=0; tIdx<nDistThreads;  ++tIdx) {
        for (unsigned int pIdx=0; pIdx<pCounts[tIdx]; ++pIdx) {
            fptr << tIdx << ',' << pIdx << ','
                 << bCounts[tIdx][pIdx] << ','
                 << wtimesPack_sec[tIdx][pIdx] << ','
                 << wtimesAsync_sec[tIdx][pIdx] << ','
                 << wtimesPacket_sec[tIdx][pIdx] << "\n";
        }
    }
    fptr.close();

    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime] End single GPU action (Timed)");
}
#    endif
#  endif
#endif

/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeCpuGpuTasks(const std::string& bundleName,
                                 const RuntimeAction& cpuAction,
                                 const TileWrapper& tilePrototype,
                                 const RuntimeAction& gpuAction,
                                 const DataPacket& packetPrototype) {
    Logger::instance().log("[Runtime] Start CPU/GPU action bundle");

    if        (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] "
                               "Given GPU action should run on packet of blocks, "
                               "which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuTasks] "
                                    "Need at least one tile per GPU packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] "
                               "Need at least two ThreadTeams in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    //
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*        cpuTeam = teams_[0];
    ThreadTeam*        gpuTeam = teams_[1];

    // Assume for no apparent reason that the GPU will finish first
    gpuTeam->attachThreadReceiver(cpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =   cpuAction.nInitialThreads
                                 + gpuAction.nInitialThreads
                                 + 1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    RuntimeBackend&                   backend = RuntimeBackend::instance();
    std::shared_ptr<Tile>             tile_cpu{};
    std::shared_ptr<Tile>             tile_gpu{};
    std::shared_ptr<DataPacket>       packet_gpu = packetPrototype.clone();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        // If we create a first shared_ptr and enqueue it with one team, it is
        // possible that this shared_ptr could have the action applied to its
        // data and go out of scope before we create a second shared_ptr.  In
        // this case, the data item's resources would be released prematurely.
        // To avoid this, we create all copies up front and before enqueing any
        // copy.
        tile_cpu = ti->buildCurrentTile();
        tile_gpu = tile_cpu;
        if ((tile_cpu.get() != tile_gpu.get()) || (tile_cpu.use_count() != 2)) {
            throw std::logic_error("[Runtime::executeCpuGpuTasks] Ownership not shared");
        }

        packet_gpu->addTile( std::move(tile_gpu) );
        if ((tile_gpu != nullptr) || (tile_gpu.use_count() != 0)) {
            throw std::logic_error("[Runtime::executeCpuGpuTasks] tile_gpu ownership not transferred");
        } else if (tile_cpu.use_count() != 2) {
            throw std::logic_error("[Runtime::executeCpuGpuTasks] Ownership not shared after transfer");
        }

        // CPU action parallel pipeline
        cpuTeam->enqueue( tilePrototype.clone( std::move(tile_cpu) ) );
        if ((tile_cpu != nullptr) || (tile_cpu.use_count() != 0)) {
            throw std::logic_error("[Runtime::executeCpuGpuTasks] tile_cpu ownership not transferred");
        }

        // GPU action parallel pipeline
        if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
            packet_gpu->pack();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));

            gpuTeam->enqueue( std::move(packet_gpu) );
            if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
                throw std::logic_error("[Runtime::executeCpuGpuTasks] packet_gpu ownership not transferred");
            }

            packet_gpu = packetPrototype.clone();
        }
    }

    if (packet_gpu->nTiles() > 0) {
        packet_gpu->pack();
        backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
        gpuTeam->enqueue( std::move(packet_gpu) );
    } else {
        packet_gpu.reset();
    }

    if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] packet_gpu ownership not transferred (after)");
    }

    gpuTeam->closeQueue(nullptr);
    cpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread 
    // in the host team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime] End CPU/GPU action");
}
#  endif
#  ifndef RUNTIME_MUST_USE_TILEITER
void Runtime::setupPipelineForCpuGpuTasks(const std::string& bundleName,
                              const RuntimeAction& gpuAction,
                              const RuntimeAction& cpuAction,
                              const DataPacket& packetPrototype) {

    Logger::instance().log("[Runtime] Start setting up CPU/GPU action");

    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime:setupPipelineForCpuGpuTasks] "
                               "Given GPU action should run on "
                               "data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime:setupPipelineForCpuGpuTasks] "
                                    "Need at least one block per packet");
    } else if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::setupPipelineForCpuGpuTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::setupPipelineForCpuGpuTasks] "
                                    "CPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime:setupPipelineForCpuGpuTasks] "
                               "Need at least two ThreadTeams in runtime");
    }
    nTilesPerPacket_ = gpuAction.nTilesPerPacket;

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    //
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*        cpuTeam = teams_[0];
    ThreadTeam*        gpuTeam = teams_[1];

    // Assume for no apparent reason that the GPU will finish first
    gpuTeam->attachThreadReceiver(cpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =   cpuAction.nInitialThreads
                                 + gpuAction.nInitialThreads
                                 + 1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::setupPipelineForCpuGpuTasks] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    gpuToHost1_.startCycle();

    packet_gpu_ = packetPrototype.clone();

    Logger::instance().log("[Runtime] End setting up CPU/GPU action");
}
void Runtime::pushTileToCpuGpuPipeline(const std::string& bundleName,
                                       const TileWrapper& tilePrototype,
                                       const DataPacket& packetPrototype,
                                       const FlashxrTileRawPtrs& tP,
                                       const FlashxTileRawInts& tI,
                                       const FlashxTileRawReals& tR) {

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Push single tile task to CPU/GPU pipeline");
#endif

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:pushTileToCpuGpuPipeline] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime:pushTileToCpuGpuPipeline] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLED THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    //
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       cpuTeam   = teams_[0];
    ThreadTeam*       gpuTeam   = teams_[1];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    std::shared_ptr<Tile>             tile_cpu{};
    std::shared_ptr<Tile>             tile_gpu{};
    {
        // If we create a first shared_ptr and enqueue it with one team, it is
        // possible that this shared_ptr could have the action applied to its
        // data and go out of scope before we create a second shared_ptr.  In
        // this case, the data item's resources would be released prematurely.
        // To avoid this, we create all copies up front and before enqueing any
        // copy.
        //tile_cpu = std::unique_ptr<Tile>{new TileFlashxr{tP, tI, tR}};
        tile_cpu = static_cast<std::shared_ptr<Tile> >(std::make_unique<TileFlashxr>(tP, tI, tR) );
        tile_gpu = tile_cpu;
        if ((tile_cpu.get() != tile_gpu.get()) || (tile_cpu.use_count() != 2)) {
            throw std::logic_error("[Runtime::pushTileToCpuGpuPipeline] Ownership not shared");
        }

        packet_gpu_->addTile( std::move(tile_gpu) );
        if ((tile_gpu != nullptr) || (tile_gpu.use_count() != 0)) {
            throw std::logic_error("[Runtime::pushTileToCpuGpuPipeline] tile_gpu ownership not transferred");
        } else if (tile_cpu.use_count() != 2) {
            throw std::logic_error("[Runtime::pushTileToCpuGpuPipeline] Ownership not shared after transfer");
        }

        // CPU action parallel pipeline
        cpuTeam->enqueue( tilePrototype.clone( std::move(tile_cpu) ) );
        if ((tile_cpu != nullptr) || (tile_cpu.use_count() != 0)) {
            throw std::logic_error("[Runtime::pushTileToCpuGpuPipeline] tile_cpu ownership not transferred");
        }

        // GPU action parallel pipeline
            if (packet_gpu_->nTiles() >= nTilesPerPacket_) {
                packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
                Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " tiles...");
#endif
                backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));

                gpuTeam->enqueue( std::move(packet_gpu_) );

#ifdef EXTRA_DEBUG
                if ((packet_gpu_ != nullptr) || (packet_gpu_.use_count() != 0)) {
                  throw std::logic_error("[Runtime::pushTileToCpuGpuPipeline] packet_gpu ownership not transferred");
                }
#endif
                packet_gpu_ = packetPrototype.clone();
            }
    }

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Single tile task was pushed to CPU/GPU pipeline");
#endif
}
void Runtime::teardownPipelineForCpuGpuTasks(const std::string& bundleName) {

    Logger::instance().log("[Runtime] Tear Down CPU/GPU action");

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:teardownPipelineForCpuGpuTasks] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime:teardownPipelineForCpuGpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    ThreadTeam*       cpuTeam   = teams_[0];
    ThreadTeam*       gpuTeam   = teams_[1];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    {

        if (packet_gpu_->nTiles() > 0) {
            packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
            Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " final tiles...");
#endif
            backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));
            gpuTeam->enqueue( std::move(packet_gpu_) );
        } else {
            packet_gpu_.reset();
        }
#ifdef EXTRA_DEBUG
        if ((packet_gpu_ != nullptr) || (packet_gpu_.use_count() != 0)) {
          throw std::logic_error("[Runtime::teardownPipelineForCpuGpuTasks] packet_gpu_ ownership not transferred (after)");
        }
#endif

    }
    gpuTeam->closeQueue(nullptr);
    cpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread 
    // in the host team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime:teardownPipelineForCpuGpuTasks] End CPU/GPU action");
}

#  endif
#endif

/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeExtendedGpuTasks(const std::string& bundleName,
                                      const unsigned int nDistributorThreads,
                                      const RuntimeAction& gpuAction,
                                      const RuntimeAction& postGpuAction,
                                      const DataPacket& packetPrototype,
                                      const TileWrapper& tilePrototype) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif
    Logger::instance().log("[Runtime] Start GPU/Post-GPU action bundle");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads";
    Logger::instance().log(msg);

    if        (nDistThreads <= 0) {
        throw std::invalid_argument("[Runtime::executeExtendedGpuTasks] "
                                    "nDistributorThreads must be positive");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeExtendedGpuTasks] "
                               "Given GPU action should run on packet of blocks, "
                               "which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeExtendedGpuTasks] "
                                    "Need at least one tile per GPU packet");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeExtendedGpuTasks] "
                               "Given post-GPU action should run on tiles, "
                               "which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeExtendedGpuTasks] "
                                    "Post-GPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime::executeExtendedGpuTasks] "
                               "Need at least two ThreadTeams in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // GPU/Post-GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU,
    //    copies results to Grid data structures, and
    //    pushes blocks to Post-GPU team
    // 4) Post-GPU action applied by host via Post-GPU team
    ThreadTeam*        gpuTeam     = teams_[0];
    ThreadTeam*        postGpuTeam = teams_[1];

    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost1_);
    gpuToHost1_.attachDataReceiver(postGpuTeam);
    gpuToHost1_.setReceiverPrototype(&tilePrototype);

    unsigned int nTotalThreads =   gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads
                                 + nDistThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeExtendedGpuTasks] "
                                "Post-GPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    postGpuTeam->startCycle(postGpuAction, "Post_GPU_Block_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    RuntimeBackend&                   backend = RuntimeBackend::instance();
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, backend, level, packetPrototype, gpuTeam, gpuAction) \
                     num_threads(nDistThreads)
#endif
    {
        std::shared_ptr<DataPacket>       packet_gpu = packetPrototype.clone();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            packet_gpu->addTile( ti->buildCurrentTile() );

            if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
                packet_gpu->pack();
                backend.initiateHostToGpuTransfer(*(packet_gpu.get()));

                gpuTeam->enqueue( std::move(packet_gpu) );

                packet_gpu = packetPrototype.clone();
            }
        }

        if (packet_gpu->nTiles() > 0) {
            packet_gpu->pack();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            gpuTeam->enqueue( std::move(packet_gpu) );
        } else {
            packet_gpu.reset();
        }

        // host thread blocks until cycle ends, so activate a thread
        gpuTeam->increaseThreadCount(1);
    }

    gpuTeam->closeQueue(nullptr);
    postGpuTeam->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();
    gpuToHost1_.detachDataReceiver();

    Logger::instance().log("[Runtime] End GPU/Post-GPU action bundle");
}
#  endif
#  ifndef RUNTIME_MUST_USE_TILEITER
void Runtime::setupPipelineForExtGpuTasks(const std::string& bundleName,
                              const RuntimeAction& gpuAction,
                              const RuntimeAction& postGpuAction,
                              const DataPacket& packetPrototype,
                              const TileWrapper& tilePrototype) {

    Logger::instance().log("[Runtime] Start setting up GPU/Post-GPU action");

    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime:setupPipelineForExtGpuTasks] "
                               "Given GPU action should run on "
                               "data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime:setupPipelineForExtGpuTasks] "
                                    "Need at least one block per packet");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::setupPipelineForExtGpuTasks] "
                               "Given post-GPU action should run on tiles, "
                               "which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::setupPipelineForExtGpuTasks] "
                                    "Post-GPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime:setupPipelineForExtGpuTasks] "
                               "Need at least two ThreadTeams in runtime");
    }
    nTilesPerPacket_ = gpuAction.nTilesPerPacket;

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // GPU/Post-GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU,
    //    copies results to Grid data structures, and
    //    pushes blocks to Post-GPU team
    // 4) Post-GPU action applied by host via Post-GPU team
    ThreadTeam*       gpuTeam   = teams_[0];
    ThreadTeam*        postGpuTeam = teams_[1];

    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost1_);
    gpuToHost1_.attachDataReceiver(postGpuTeam);
    gpuToHost1_.setReceiverPrototype(&tilePrototype);

    unsigned int nTotalThreads =   gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads
                                 + 1;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::setupPipelineForExtGpuTasks] "
                                "Post-GPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    postGpuTeam->startCycle(postGpuAction, "Post_GPU_Block_Team");
    gpuToHost1_.startCycle();

    {
        packet_gpu_ = packetPrototype.clone();
    }

    Logger::instance().log("[Runtime] End setting up GPU/Post-GPU action");
}
void Runtime::pushTileToExtGpuPipeline(const std::string& bundleName,
                                       const DataPacket& packetPrototype,
                                       const FlashxrTileRawPtrs& tP,
                                       const FlashxTileRawInts& tI,
                                       const FlashxTileRawReals& tR) {

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Push single tile task to GPU/Post-GPU pipeline");
#endif

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:pushTileToExtGpuPipeline] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime:pushTileToExtGpuPipeline] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLED THREAD TEAM CONFIGURATION
    // GPU/Post-GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU,
    //    copies results to Grid data structures, and
    //    pushes blocks to Post-GPU team
    // 4) Post-GPU action applied by host via Post-GPU team
    ThreadTeam*       gpuTeam   = teams_[0];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    {
      //            packet_gpu_->addTile( std::unique_ptr<Tile>{new TileFlashxr{tP, tI, tR}} );
                  packet_gpu_->addTile( static_cast<std::shared_ptr<Tile> >(std::make_unique<TileFlashxr>(tP, tI, tR) ));
                  //      packet_gpu_->addTile( std::make_shared<TileFlashxr>(tP, tI, tR) );
            if (packet_gpu_->nTiles() >= nTilesPerPacket_) {
                packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
                Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " tiles...");
#endif
                backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));

                gpuTeam->enqueue( std::move(packet_gpu_) );

                packet_gpu_ = packetPrototype.clone();
            }
    }

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Single tile task was pushed to GPU/Post-GPU pipeline");
#endif
}
void Runtime::teardownPipelineForExtGpuTasks(const std::string& bundleName) {

    Logger::instance().log("[Runtime] Tear Down GPU/Post-GPU action");

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:teardownPipelineForExtGpuTasks] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime:teardownPipelineForExtGpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       gpuTeam   = teams_[0];
    ThreadTeam*        postGpuTeam = teams_[1];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    {

        if (packet_gpu_->nTiles() > 0) {
            packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
            Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " final tiles...");
#endif
            backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));
            gpuTeam->enqueue( std::move(packet_gpu_) );
        } else {
            packet_gpu_.reset();
        }
        // host thread blocks until cycle ends, so activate a thread
        gpuTeam->increaseThreadCount(1);
    }
    gpuTeam->closeQueue(nullptr);
    postGpuTeam->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();
    gpuToHost1_.detachDataReceiver();

    Logger::instance().log("[Runtime:teardownPipelineForExtGpuTasks] End GPU/Post-GPU action");
}

#  endif
#endif

/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeCpuGpuSplitTasks(const std::string& bundleName,
                                      const unsigned int nDistributorThreads,
                                      const unsigned int stagger_usec,
                                      const RuntimeAction& cpuAction,
                                      const TileWrapper& tilePrototype,
                                      const RuntimeAction& gpuAction,
                                      const DataPacket& packetPrototype,
                                      const unsigned int nTilesPerCpuTurn) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif
    Logger::instance().log("[Runtime] Start CPU/GPU shared action");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads / stagger of " 
                        + std::to_string(stagger_usec)
                        + " us";
    Logger::instance().log(msg);
    msg = "[Runtime] "
         + std::to_string(nTilesPerCpuTurn)
         + " tiles sent to CPU for every packet of "
         + std::to_string(gpuAction.nTilesPerPacket)
         + " tiles sent to GPU";
    Logger::instance().log(msg);

    if        (nDistThreads <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks] "
                                    "nDistributorThreads must be positive");
    } else if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                               "Given GPU action should run on packet of blocks, "
                               "which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks] "
                                    "Need at least one tile per GPU packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                               "Need at least two ThreadTeams in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU/GPU action parallel pipeline
    // 1) Action Parallel Distributor will send one fraction of data items
    //    to CPU for computation
    // 2) For the remaining data items,
    //    a) Asynchronous transfer of Packets of Blocks to GPU by distributor,
    //    b) GPU action applied to blocks in packet by GPU team
    //    c) Mover/Unpacker transfers packet back to CPU and
    //       copies results to Grid data structures
    ThreadTeam*        cpuTeam = teams_[0];
    ThreadTeam*        gpuTeam = teams_[1];

    // Assume that the GPU task function is heavy enough that the GPU team's
    // threads will usually sleep and therefore not battle the host-side
    // computation threads for resources.  Based on this, we concentrate on
    // getting the CPU teams as many threads as possible (i.e. by setting a high
    // initial thread number and by giving distributor threads to the team) and
    // will let the GPU threads go to sleep once the GPU work is done.  Simpler
    // and more predictable host-side thread balancing.
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =   cpuAction.nInitialThreads
                                 + nDistThreads;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "ActionSharing_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "ActionSharing_GPU_Packet_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    RuntimeBackend&                   backend = RuntimeBackend::instance();
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, backend, level, packetPrototype, \
                     cpuTeam, gpuTeam, gpuAction, \
                     stagger_usec, nTilesPerCpuTurn) \
                     num_threads(nDistThreads)
#endif
    {
#ifdef USE_THREADED_DISTRIBUTOR
        int         tId = omp_get_thread_num();
#else
        int         tId = 0;
#endif
        bool        isCpuTurn = true;
        int         nInCpuTurn = 0;

        std::this_thread::sleep_for(std::chrono::microseconds(tId * stagger_usec));

        std::shared_ptr<Tile>             tileDesc{};
        std::shared_ptr<DataPacket>       packet_gpu = packetPrototype.clone();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();

            if (isCpuTurn) {
                cpuTeam->enqueue( tilePrototype.clone( std::move(tileDesc) ) );

                ++nInCpuTurn;
                if (nInCpuTurn >= nTilesPerCpuTurn) {
                    isCpuTurn = false;
                    nInCpuTurn = 0;
                }
            } else {
                packet_gpu->addTile( std::move(tileDesc) );

                if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
                    packet_gpu->pack();
                    backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
                    gpuTeam->enqueue( std::move(packet_gpu) );

                    packet_gpu = packetPrototype.clone();
                    isCpuTurn = true;
                }
            }
        }

        if (packet_gpu->nTiles() > 0) {
            packet_gpu->pack();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            gpuTeam->enqueue( std::move(packet_gpu) );
        } else {
            packet_gpu.reset();
        }

        cpuTeam->increaseThreadCount(1);
    } // implied barrier
    gpuTeam->closeQueue(nullptr);
    cpuTeam->closeQueue(nullptr);

    cpuTeam->wait();
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime] End CPU/GPU shared action");
}
#  endif
#  ifndef RUNTIME_MUST_USE_TILEITER
void Runtime::setupPipelineForCpuGpuSplitTasks(const std::string& bundleName,
                                               const unsigned int stagger_usec,
                                               const RuntimeAction& gpuAction,
                                               const RuntimeAction& cpuAction,
                                               const DataPacket& packetPrototype,
                                               const unsigned int nTilesPerCpuTurn) {

    Logger::instance().log("[Runtime] Start setting up CPU/GPU shared action");
    std::string   msg =   "[Runtime] "
         + std::to_string(nTilesPerCpuTurn)
         + " tiles sent to CPU for every packet of "
         + std::to_string(gpuAction.nTilesPerPacket)
         + " tiles sent to GPU";
    Logger::instance().log(msg);

    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime:setupPipelineForCpuGpuSplitTasks] "
                               "Given GPU action should run on "
                               "data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime:setupPipelineForCpuGpuSplitTasks] "
                                    "Need at least one block per packet");
    } else if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::setupPipelineForCpuGpuSplitTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::setupPipelineForCpuGpuSplitTasks] "
                                    "CPU action should have zero tiles/packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime:setupPipelineForCpuGpuSplitTasks] "
                               "Need at least two ThreadTeams in runtime");
    }
    nTilesPerPacket_ = gpuAction.nTilesPerPacket;
    nTilesPerCpuTurn_ = nTilesPerCpuTurn;
    isCpuTurn_ = true;
    nInCpuTurn_ = 0;

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU/GPU action parallel pipeline
    // 1) Action Parallel Distributor will send one fraction of data items
    //    to CPU for computation
    // 2) For the remaining data items,
    //    a) Asynchronous transfer of Packets of Blocks to GPU by distributor,
    //    b) GPU action applied to blocks in packet by GPU team
    //    c) Mover/Unpacker transfers packet back to CPU and
    //       copies results to Grid data structures
    ThreadTeam*        cpuTeam = teams_[0];
    ThreadTeam*        gpuTeam = teams_[1];

    // Assume that the GPU task function is heavy enough that the GPU team's
    // threads will usually sleep and therefore not battle the host-side
    // computation threads for resources.  Based on this, we concentrate on
    // getting the CPU teams as many threads as possible (i.e. by setting a high
    // initial thread number and by giving distributor threads to the team) and
    // will let the GPU threads go to sleep once the GPU work is done.  Simpler
    // and more predictable host-side thread balancing.
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =   cpuAction.nInitialThreads
                                 + 1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::setupPipelineForCpuGpuSplitTasks] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "ActionSharing_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "ActionSharing_GPU_Packet_Team");
    gpuToHost1_.startCycle();

    packet_gpu_ = packetPrototype.clone();

    Logger::instance().log("[Runtime] End setting up CPU/GPU shared action");
}
void Runtime::pushTileToCpuGpuSplitPipeline(const std::string& bundleName,
                                            const TileWrapper& tilePrototype,
                                            const DataPacket& packetPrototype,
                                            const FlashxrTileRawPtrs& tP,
                                            const FlashxTileRawInts& tI,
                                            const FlashxTileRawReals& tR) {

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Push single tile task to CPU/GPU split pipeline");
#endif

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:pushTileToCpuGpuSplitPipeline] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime:pushTileToCpuGpuSplitPipeline] "
                               "Need at least one ThreadTeam in runtime");
    }

    //***** ASSEMBLED THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    //
    // GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*       cpuTeam   = teams_[0];
    ThreadTeam*       gpuTeam   = teams_[1];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    std::shared_ptr<Tile>             tileDesc{};
    {
#ifdef USE_THREADED_DISTRIBUTOR
        int         tId = omp_get_thread_num();
        std::this_thread::sleep_for(std::chrono::microseconds(tId * stagger_usec));
#endif

        // If we create a first shared_ptr and enqueue it with one team, it is
        // possible that this shared_ptr could have the action applied to its
        // data and go out of scope before we create a second shared_ptr.  In
        // this case, the data item's resources would be released prematurely.
        // To avoid this, we create all copies up front and before enqueing any
        // copy.
        //tileDesc = std::unique_ptr<Tile>{new TileFlashxr{tP, tI, tR}};
        tileDesc = static_cast<std::shared_ptr<Tile> >(std::make_unique<TileFlashxr>(tP, tI, tR) );

        if (isCpuTurn_) {
                cpuTeam->enqueue( tilePrototype.clone( std::move(tileDesc) ) );
                if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
                  throw std::logic_error("[Runtime::pushTileToCpuGpuSplitPipeline] tileDesc ownership not transferred");
                }

                ++nInCpuTurn_;
                if (nInCpuTurn_ >= nTilesPerCpuTurn_) {
                    isCpuTurn_ = false;
                    nInCpuTurn_ = 0;
                }
        } else {
                packet_gpu_->addTile( std::move(tileDesc) );
                if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
                  throw std::logic_error("[Runtime::pushTileToCpuGpuSplitPipeline] tileDesc ownership not transferred");
                }

                if (packet_gpu_->nTiles() >= nTilesPerPacket_) {
                    packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
                    Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " tiles...");
#endif
                    backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));
                    gpuTeam->enqueue( std::move(packet_gpu_) );

                    packet_gpu_ = packetPrototype.clone();
                    isCpuTurn_ = true;
                }
        }
    }

#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Single tile task was pushed to CPU/GPU split pipeline");
#endif
}
void Runtime::teardownPipelineForCpuGpuSplitTasks(const std::string& bundleName) {

    Logger::instance().log("[Runtime] Tear Down CPU/GPU shared action");

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:teardownPipelineForCpuGpuSplitTasks] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime:teardownPipelineForCpuGpuSplitTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    ThreadTeam*       cpuTeam   = teams_[0];
    ThreadTeam*       gpuTeam   = teams_[1];

    RuntimeBackend&               backend = RuntimeBackend::instance();
    {

        if (packet_gpu_->nTiles() > 0) {
            packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
            Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " final tiles...");
#endif
            backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));
            gpuTeam->enqueue( std::move(packet_gpu_) );
        } else {
            packet_gpu_.reset();
        }
#ifdef EXTRA_DEBUG
        if ((packet_gpu_ != nullptr) || (packet_gpu_.use_count() != 0)) {
          throw std::logic_error("[Runtime::teardownPipelineForCpuGpuSplitTasks] packet_gpu_ ownership not transferred (after)");
        }
#endif

    }
    gpuTeam->closeQueue(nullptr);
    cpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread
    // in the host team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime:teardownPipelineForCpuGpuSplitTasks] End CPU/GPU shared action");
}

#  endif
#endif


/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef MILHOJA_TIMED_PIPELINE_CONFIGS
#    ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeCpuGpuSplitTasks_timed(const std::string& bundleName,
                                            const unsigned int nDistributorThreads,
                                            const unsigned int stagger_usec,
                                            const RuntimeAction& cpuAction,
                                            const TileWrapper& tilePrototype,
                                            const RuntimeAction& gpuAction,
                                            const DataPacket& packetPrototype,
                                            const unsigned int nTilesPerCpuTurn,
                                            const unsigned int stepNumber,
                                            const MPI_Comm comm) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif
    Logger::instance().log("[Runtime] Start CPU/GPU shared action (Timed)");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads / stagger of " 
                        + std::to_string(stagger_usec)
                        + " us";
    Logger::instance().log(msg);
    msg = "[Runtime] "
         + std::to_string(nTilesPerCpuTurn)
         + " tiles sent to CPU for every packet of "
         + std::to_string(gpuAction.nTilesPerPacket)
         + " tiles sent to GPU";
    Logger::instance().log(msg);

    if        (nDistThreads <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks_timed] "
                                    "nDistributorThreads must be positive");
    } else if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks_timed] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks_timed] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks_timed] "
                               "Given GPU action should run on packet of blocks, "
                               "which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks_timed] "
                                    "Need at least one tile per GPU packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks_timed] "
                               "Need at least two ThreadTeams in runtime");
    }

    Grid&             grid = Grid::instance();
    RuntimeBackend&   backend = RuntimeBackend::instance();

    //***** SETUP TIMING
    int rank = -1;
    MPI_Comm_rank(comm, &rank);

    unsigned int   nxb = 1;
    unsigned int   nyb = 1;
    unsigned int   nzb = 1;
    grid.getBlockSize(&nxb, &nyb, &nzb);

    unsigned int  nPackets = ceil(  (double)grid.getNumberLocalBlocks()
                                  / (double)gpuAction.nTilesPerPacket);

    unsigned int  pCounts[nDistThreads];
    unsigned int  bCounts[nDistThreads][nPackets];
    double        wtimesPack_sec[nDistThreads][nPackets];
    double        wtimesAsync_sec[nDistThreads][nPackets];
    double        wtimesPacket_sec[nDistThreads][nPackets];

    std::string   filename("timings_packet_step");
    filename += std::to_string(stepNumber);
    filename += "_rank";
    filename += std::to_string(rank);
    filename += ".dat";

    std::ofstream   fptr;
    fptr.open(filename, std::ios::out);
    fptr << "# Testname = Data Parallel CPU/GPU\n";
    fptr << "# Step = " << stepNumber << "\n";
    fptr << "# MPI rank = " << rank << "\n";
    fptr << "# Dimension = " << MILHOJA_NDIM << "\n";
    fptr << "# NXB = " << nxb << "\n";
    fptr << "# NYB = " << nyb << "\n";
    fptr << "# NZB = " << nzb << "\n";
    fptr << "# n_distributor_threads = " << nDistThreads << "\n";
    fptr << "# stagger_usec = " << stagger_usec << "\n";
    fptr << "# n_cpu_threads = " << cpuAction.nInitialThreads << "\n";
    fptr << "# n_gpu_threads = " << gpuAction.nInitialThreads << "\n";
    fptr << "# n_blocks_per_packet = " << gpuAction.nTilesPerPacket << "\n";
    fptr << "# n_blocks_per_cpu_turn = " << nTilesPerCpuTurn << "\n";
    fptr << "# MPI_Wtick_sec = " << MPI_Wtick() << "\n";
    fptr << "# thread,packet,nblocks,walltime_pack_sec,walltime_async_sec,walltime_packet_sec\n";

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU/GPU action parallel pipeline
    // 1) Action Parallel Distributor will send one fraction of data items
    //    to CPU for computation
    // 2) For the remaining data items,
    //    a) Asynchronous transfer of Packets of Blocks to GPU by distributor,
    //    b) GPU action applied to blocks in packet by GPU team
    //    c) Mover/Unpacker transfers packet back to CPU and
    //       copies results to Grid data structures
    ThreadTeam*        cpuTeam = teams_[0];
    ThreadTeam*        gpuTeam = teams_[1];

    // Assume that the GPU task function is heavy enough that the GPU team's
    // threads will usually sleep and therefore not battle the host-side
    // computation threads for resources.  Based on this, we concentrate on
    // getting the CPU teams as many threads as possible (i.e. by setting a high
    // initial thread number and by giving distributor threads to the team) and
    // will let the GPU threads go to sleep once the GPU work is done.  Simpler
    // and more predictable host-side thread balancing.
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =   cpuAction.nInitialThreads
                                 + nDistThreads;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks_timed] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "ActionSharing_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "ActionSharing_GPU_Packet_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int   level = 0;
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, backend, level, packetPrototype, \
                            cpuTeam, gpuTeam, gpuAction, nTilesPerCpuTurn, \
                            stagger_usec, \
                            wtimesPack_sec, wtimesAsync_sec, wtimesPacket_sec, \
                            pCounts, bCounts) \
                     num_threads(nDistThreads)
#endif
    {
#ifdef USE_THREADED_DISTRIBUTOR
        int         tIdx = omp_get_thread_num();
#else
        int         tIdx = 0;
#endif
        bool        isCpuTurn = true;
        int         nInCpuTurn = 0;

        unsigned int  pIdx         = 0;
        double        tStartPacket = 0.0;
        double        tStartPack   = 0.0;
        double        tStartAsync  = 0.0;

        std::shared_ptr<Tile>             tileDesc{};

        std::this_thread::sleep_for(std::chrono::microseconds(tIdx * stagger_usec));

        tStartPacket = MPI_Wtime();
        std::shared_ptr<DataPacket>       packet_gpu = packetPrototype.clone();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();

            if (isCpuTurn) {
                cpuTeam->enqueue( tilePrototype.clone( std::move(tileDesc) ) );

                ++nInCpuTurn;
                if (nInCpuTurn >= nTilesPerCpuTurn) {
                    isCpuTurn = false;
                    nInCpuTurn = 0;
                }
            } else {
                packet_gpu->addTile( std::move(tileDesc) );

                if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
                    tStartPack = MPI_Wtime();
                    packet_gpu->pack();
                    wtimesPack_sec[tIdx][pIdx] = MPI_Wtime() - tStartPack;

                    tStartAsync = MPI_Wtime();
                    backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
                    wtimesAsync_sec[tIdx][pIdx] = MPI_Wtime() - tStartAsync;

                    bCounts[tIdx][pIdx] = packet_gpu->nTiles();
                    gpuTeam->enqueue( std::move(packet_gpu) );

                    wtimesPacket_sec[tIdx][pIdx] = MPI_Wtime() - tStartPacket;

                    ++pIdx;
                    isCpuTurn = true;

                    tStartPacket = MPI_Wtime();
                    packet_gpu = packetPrototype.clone();
                }
            }
        }

        if (packet_gpu->nTiles() > 0) {
            tStartPack = MPI_Wtime();
            packet_gpu->pack();
            wtimesPack_sec[tIdx][pIdx] = MPI_Wtime() - tStartPack;

            tStartAsync = MPI_Wtime();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            wtimesAsync_sec[tIdx][pIdx] = MPI_Wtime() - tStartAsync;

            bCounts[tIdx][pIdx] = packet_gpu->nTiles();
            gpuTeam->enqueue( std::move(packet_gpu) );

            wtimesPacket_sec[tIdx][pIdx] = MPI_Wtime() - tStartPacket;

            ++pIdx;
        } else {
            packet_gpu.reset();
        }
        pCounts[tIdx] = pIdx;

        cpuTeam->increaseThreadCount(1);
    } // implied barrier
    gpuTeam->closeQueue(nullptr);
    cpuTeam->closeQueue(nullptr);

    fptr << std::setprecision(15);
    for     (unsigned int tIdx=0; tIdx<nDistThreads;  ++tIdx) {
        for (unsigned int pIdx=0; pIdx<pCounts[tIdx]; ++pIdx) {
            fptr << tIdx << ',' << pIdx << ','
                 << bCounts[tIdx][pIdx] << ','
                 << wtimesPack_sec[tIdx][pIdx] << ','
                 << wtimesAsync_sec[tIdx][pIdx] << ','
                 << wtimesPacket_sec[tIdx][pIdx] << "\n";
        }
    }
    fptr.close();

    cpuTeam->wait();
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime] End CPU/GPU shared action (Timed)");
}
#    endif
#  endif
#endif

/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeExtendedCpuGpuSplitTasks(const std::string& bundleName,
                                              const unsigned int nDistributorThreads,
                                              const RuntimeAction& actionA_cpu,
                                              const TileWrapper& tilePrototype,
                                              const RuntimeAction& actionA_gpu,
                                              const DataPacket& packetPrototype,
                                              const RuntimeAction& postActionB_cpu,
                                              const TileWrapper& postTilePrototype,
                                              const unsigned int nTilesPerCpuTurn) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif
    Logger::instance().log("[Runtime] Start extended CPU/GPU shared action");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads";
    Logger::instance().log(msg);
    msg =   "[Runtime] "
          + std::to_string(nTilesPerCpuTurn)
          + " tiles sent to CPU for every packet of "
          + std::to_string(actionA_gpu.nTilesPerPacket)
          + " tiles sent to GPU";
    Logger::instance().log(msg);

    if        (nDistThreads <= 0) {
        throw std::invalid_argument("[Runtime::executeExtendedCpuGpuSplitTasks] "
                                    "nDistributorThreads must be positive");
    } else if (actionA_cpu.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeExtendedCpuGpuSplitTasks] "
                               "Given CPU action A should run on tiles, "
                               "which is not in configuration");
    } else if (actionA_cpu.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeExtendedCpuGpuSplitTasks] "
                                    "CPU A tiles/packet should be zero since it is tile-based");
    } else if (actionA_gpu.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeExtendedCpuGpuSplitTasks] "
                               "Given GPU action should run on packet of blocks, "
                               "which is not in configuration");
    } else if (actionA_gpu.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeExtendedCpuGpuSplitTasks] "
                                    "Need at least one tile per GPU packet");
    } else if (postActionB_cpu.teamType != actionA_cpu.teamType) {
        throw std::logic_error("[Runtime::executeExtendedCpuGpuSplitTasks] "
                               "Given post action data type must match that "
                               "of CPU action A");
    } else if (postActionB_cpu.nTilesPerPacket != actionA_cpu.nTilesPerPacket) {
        throw std::invalid_argument("[Runtime::executeExtendedCpuGpuSplitTasks] "
                                    "Given post action tiles/packet must match that "
                                    "of CPU action A");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[Runtime::executeExtendedCpuGpuSplitTasks] "
                               "Need at least three ThreadTeams in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU/GPU action parallel pipeline
    // 1) Action Parallel Distributor will send one fraction of data items
    //    to CPU for computation and each of these is enqueued directly with the post
    //    action thread team.
    // 2) For the remaining data items,
    //    a) Asynchronous transfer of Packets of Blocks to GPU by distributor,
    //    b) GPU action applied to blocks in packet by GPU team
    //    c) Mover/Unpacker transfers packet back to CPU,
    //       copies results to Grid data structures,
    //       and enqueues with post action thread team.
    ThreadTeam*        teamA_cpu = teams_[0];
    ThreadTeam*        teamA_gpu = teams_[1];
    ThreadTeam*        teamB_cpu = teams_[2];

    teamA_cpu->attachThreadReceiver(teamB_cpu);
    teamA_cpu->attachDataReceiver(teamB_cpu);
    teamA_cpu->setReceiverPrototype(&postTilePrototype);
    teamA_gpu->attachDataReceiver(&gpuToHost1_);
    gpuToHost1_.attachDataReceiver(teamB_cpu);
    gpuToHost1_.setReceiverPrototype(&postTilePrototype);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =   actionA_cpu.nInitialThreads
                                 + nDistThreads;
    if (nTotalThreads > teamA_cpu->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeExtendedCpuGpuSplitTasks] "
                                "CPU team could receive too many thread "
                                "activation calls");
    }
    nTotalThreads =   actionA_cpu.nInitialThreads
                    + postActionB_cpu.nInitialThreads
                    + nDistThreads;
    if (nTotalThreads > teamB_cpu->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeExtendedCpuGpuSplitTasks] "
                                "Post could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    teamA_cpu->startCycle(actionA_cpu, "ActionSharing_CPU_Block_Team");
    teamA_gpu->startCycle(actionA_gpu, "ActionSharing_GPU_Packet_Team");
    teamB_cpu->startCycle(postActionB_cpu, "PostAction_CPU_Block_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int         level = 0;
    Grid&                grid = Grid::instance();
    RuntimeBackend&      backend = RuntimeBackend::instance();
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, backend, level, packetPrototype, teamA_cpu, teamA_gpu, actionA_gpu, nTilesPerCpuTurn) \
                     num_threads(nDistThreads)
#endif
    {
#ifdef USE_THREADED_DISTRIBUTOR
        int         tId = omp_get_thread_num();
#else
        int         tId = 0;
#endif
        bool        isCpuTurn = ((tId % 2) == 0);
        int         nInCpuTurn = 0;

        std::shared_ptr<Tile>             tileDesc{};
        std::shared_ptr<DataPacket>       packet_gpu = packetPrototype.clone();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();

            if (isCpuTurn) {
                teamA_cpu->enqueue( tilePrototype.clone( std::move(tileDesc) ) );

                ++nInCpuTurn;
                if (nInCpuTurn >= nTilesPerCpuTurn) {
                    isCpuTurn = false;
                    nInCpuTurn = 0;
                }
            } else {
                packet_gpu->addTile( std::move(tileDesc) );

                if (packet_gpu->nTiles() >= actionA_gpu.nTilesPerPacket) {
                    packet_gpu->pack();
                    backend.initiateHostToGpuTransfer(*(packet_gpu.get()));

                    teamA_gpu->enqueue( std::move(packet_gpu) );

                    packet_gpu = packetPrototype.clone();
                    isCpuTurn = true;
                }
            }
        }

        if (packet_gpu->nTiles() > 0) {
            packet_gpu->pack();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            teamA_gpu->enqueue( std::move(packet_gpu) );
        } else {
            packet_gpu.reset();
        }

        teamA_cpu->increaseThreadCount(1);
    } // implied barrier
    teamA_gpu->closeQueue(nullptr);
    teamA_cpu->closeQueue(nullptr);

    // All data flowing through the Action B/Post-A team
    teamB_cpu->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    teamA_cpu->detachThreadReceiver();
    teamA_cpu->detachDataReceiver();
    teamA_gpu->detachDataReceiver();
    gpuToHost1_.detachDataReceiver();

    Logger::instance().log("[Runtime] End Extended CPU/GPU shared action");
}
#  endif
#  ifndef RUNTIME_MUST_USE_TILEITER
void Runtime::setupPipelineForExtCpuGpuSplitTasks(const std::string& bundleName,
                                                  const RuntimeAction& actionA_cpu,
                                                  const TileWrapper& tilePrototype,
                                                  const RuntimeAction& actionA_gpu,
                                                  const DataPacket& packetPrototype,
                                                  const RuntimeAction& postActionB_cpu,
                                                  const TileWrapper& postTilePrototype,
                                                  const unsigned int nTilesPerCpuTurn) {

    const unsigned int  nDistThreads = 1;

    Logger::instance().log("[Runtime] Start extended CPU/GPU shared action");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads";
    Logger::instance().log(msg);
    msg =   "[Runtime] "
          + std::to_string(nTilesPerCpuTurn)
          + " tiles sent to CPU for every packet of "
          + std::to_string(actionA_gpu.nTilesPerPacket)
          + " tiles sent to GPU";
    Logger::instance().log(msg);

    if        (nDistThreads <= 0) {
        throw std::invalid_argument("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                                    "nDistributorThreads must be positive");
    } else if (actionA_cpu.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                               "Given CPU action A should run on tiles, "
                               "which is not in configuration");
    } else if (actionA_cpu.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                                    "CPU A tiles/packet should be zero since it is tile-based");
    } else if (actionA_gpu.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                               "Given GPU action should run on packet of blocks, "
                               "which is not in configuration");
    } else if (actionA_gpu.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                                    "Need at least one tile per GPU packet");
    } else if (postActionB_cpu.teamType != actionA_cpu.teamType) {
        throw std::logic_error("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                               "Given post action data type must match that "
                               "of CPU action A");
    } else if (postActionB_cpu.nTilesPerPacket != actionA_cpu.nTilesPerPacket) {
        throw std::invalid_argument("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                                    "Given post action tiles/packet must match that "
                                    "of CPU action A");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                               "Need at least three ThreadTeams in runtime");
    }
    nTilesPerPacket_ = actionA_gpu.nTilesPerPacket;
    nTilesPerCpuTurn_ = nTilesPerCpuTurn;
    isCpuTurn_ = true;
    nInCpuTurn_ = 0;

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU/GPU action parallel pipeline
    // 1) Action Parallel Distributor will send one fraction of data items
    //    to CPU for computation and each of these is enqueued directly with the post
    //    action thread team.
    // 2) For the remaining data items,
    //    a) Asynchronous transfer of Packets of Blocks to GPU by distributor,
    //    b) GPU action applied to blocks in packet by GPU team
    //    c) Mover/Unpacker transfers packet back to CPU,
    //       copies results to Grid data structures,
    //       and enqueues with post action thread team.
    ThreadTeam*        teamA_cpu = teams_[0];
    ThreadTeam*        teamA_gpu = teams_[1];
    ThreadTeam*        teamB_cpu = teams_[2];

    teamA_cpu->attachThreadReceiver(teamB_cpu);
    teamA_cpu->attachDataReceiver(teamB_cpu);
    teamA_cpu->setReceiverPrototype(&postTilePrototype);
    teamA_gpu->attachDataReceiver(&gpuToHost1_);
    gpuToHost1_.attachDataReceiver(teamB_cpu);
    gpuToHost1_.setReceiverPrototype(&postTilePrototype);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =   actionA_cpu.nInitialThreads
                                 + nDistThreads;
    if (nTotalThreads > teamA_cpu->nMaximumThreads()) {
        throw std::logic_error("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                                "CPU team could receive too many thread "
                                "activation calls");
    }
    nTotalThreads =   actionA_cpu.nInitialThreads
                    + postActionB_cpu.nInitialThreads
                    + nDistThreads;
    if (nTotalThreads > teamB_cpu->nMaximumThreads()) {
        throw std::logic_error("[Runtime::setupPipelineForExtCpuGpuSplitTasks] "
                                "Post could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    teamA_cpu->startCycle(actionA_cpu, "ActionSharing_CPU_Block_Team");
    teamA_gpu->startCycle(actionA_gpu, "ActionSharing_GPU_Packet_Team");
    teamB_cpu->startCycle(postActionB_cpu, "PostAction_CPU_Block_Team");
    gpuToHost1_.startCycle();

    packet_gpu_ = packetPrototype.clone();

    Logger::instance().log("[Runtime] End setting up extended CPU/GPU shared action");
}

void Runtime::pushTileToExtCpuGpuSplitPipeline(const std::string& bundleName,
                                               const TileWrapper& tilePrototype,
                                               const DataPacket& packetPrototype,
                                               const TileWrapper& postTilePrototype,
                                               const FlashxrTileRawPtrs& tP,
                                               const FlashxTileRawInts& tI,
                                               const FlashxTileRawReals& tR) {
#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Push single tile task to EXT CPU/GPU split pipeline");
#endif
    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:pushTileToExtCpuGpuSplitPipeline] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[Runtime:pushTileToExtCpuGpuSplitPipeline] "
                               "Need at three ThreadTeams in runtime");
    }

    ThreadTeam*        teamA_cpu = teams_[0];
    ThreadTeam*        teamA_gpu = teams_[1];
    ThreadTeam*        teamB_cpu = teams_[2];

    RuntimeBackend&      backend = RuntimeBackend::instance();
    std::shared_ptr<Tile>             tileDesc{};
    {

        tileDesc = static_cast<std::shared_ptr<Tile>>(std::make_unique<TileFlashxr>(tP, tI, tR));
        if (isCpuTurn_) {
            teamA_cpu->enqueue( tilePrototype.clone( std::move(tileDesc) ) );
            if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
                throw std::logic_error("[Runtime::pushTileToExtCpuGpuSplitPipeline] tileDesc ownership not transferred");
            }

            ++nInCpuTurn_;
            if (nInCpuTurn_ >= nTilesPerCpuTurn_) {
                isCpuTurn_ = false;
                nInCpuTurn_ = 0;
            }
        } else {
            packet_gpu_->addTile( std::move(tileDesc) );
            if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
                throw std::logic_error("[Runtime::pushTileToExtCpuGpuSplitPipeline] tileDesc ownership not transferred");
            }

            if (packet_gpu_->nTiles() >= nTilesPerPacket_) {
                packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
                Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " tiles...");
#endif
                backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));
                teamA_gpu->enqueue( std::move(packet_gpu_) );

                packet_gpu_ = packetPrototype.clone();
                isCpuTurn_ = true;
            }
        }
    }
#ifdef RUNTIME_PERTILE_LOG
    Logger::instance().log("[Runtime] Single tile task was pushed to EXT CPU/GPU split pipeline");
#endif
}

void Runtime::teardownPipelineForExtCpuGpuSplitTasks(const std::string& bundleName) {

    Logger::instance().log("[Runtime] Tear Down extended CPU/GPU shared action");

    if (nTilesPerPacket_ <= 0) {
        throw std::invalid_argument("[Runtime:teardownPipelineForExtCpuGpuSplitTasks] "
                                    "Need at least one block per packet");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[Runtime:teardownPipelineForExtCpuGpuSplitTasks] "
                               "Need at least three ThreadTeams in runtime");
    }
    ThreadTeam*        teamA_cpu = teams_[0];
    ThreadTeam*        teamA_gpu = teams_[1];
    ThreadTeam*        teamB_cpu = teams_[2];

    RuntimeBackend&      backend = RuntimeBackend::instance();
    {
        if (packet_gpu_->nTiles() > 0) {
            packet_gpu_->pack();
#ifdef RUNTIME_PERTILE_LOG
            Logger::instance().log("[Runtime] Shipping off packet with "
                                       + std::to_string(packet_gpu_->nTiles())
                                       + " final tiles...");
#endif
            backend.initiateHostToGpuTransfer(*(packet_gpu_.get()));
            teamA_gpu->enqueue( std::move(packet_gpu_) );
        } else {
            packet_gpu_.reset();
        }

        teamA_cpu->increaseThreadCount(1);
    } // implied barrier

    teamA_gpu->closeQueue(nullptr);
    teamA_cpu->closeQueue(nullptr);

    // All data flowing through the Action B/Post-A team
    teamB_cpu->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    teamA_cpu->detachThreadReceiver();
    teamA_cpu->detachDataReceiver();
    teamA_gpu->detachDataReceiver();
    gpuToHost1_.detachDataReceiver();

    Logger::instance().log("[Runtime:teardownPipelineForExtCpuGpuSplitTasks] End extended CPU/GPU shared action");

}
#  endif   // ifndef RUNTIME_MUST_USE_TILEITER
#endif     // ifdef RUNTIME_SUPPORT_DATAPACKETS

/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeCpuGpuWowzaTasks(const std::string& bundleName,
                                      const RuntimeAction& actionA_cpu,
                                      const TileWrapper& tilePrototype,
                                      const RuntimeAction& actionA_gpu,
                                      const RuntimeAction& actionB_gpu,
                                      const DataPacket& packetPrototypeA,
                                      const DataPacket& packetPrototypeB,
                                      const unsigned int nTilesPerCpuTurn) {
    Logger::instance().log("[Runtime] Start CPU/GPU shared & GPU configuration");
    std::string   msg = "[Runtime] "
                        + std::to_string(nTilesPerCpuTurn)
                        + " tiles sent to action A CPU team for every packet of "
                        + std::to_string(actionA_gpu.nTilesPerPacket)
                        + " tiles sent to action A GPU team";
    Logger::instance().log(msg);

    if        (actionA_cpu.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeCpuGpuWowzaTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (actionA_cpu.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuWowzaTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (actionA_gpu.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeCpuGpuWowzaTasks] "
                               "Given action A GPU routine should run on packets");
    } else if (actionA_gpu.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuWowzaTasks] "
                                    "Need at least one tile per packet for action A");
    } else if (actionB_gpu.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeCpuGpuWowzaTasks] "
                               "Given action B GPU routine should run on packets");
    } else if (actionB_gpu.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuWowzaTasks] "
                                    "Need at least one tile per packet for action B");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[Runtime::executeCpuGpuWowzaTasks] "
                               "Need at least three ThreadTeams in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU/GPU action A pipeline
    // 1) Action Parallel Distributor will send one fraction of data items
    //    to CPU for computation of Action A
    // 2) For the remaining data items,
    //    a) Asynchronous transfer of Packets of Blocks to GPU by distributor,
    //    b) GPU action A applied to blocks in packet by Action A GPU team
    //    c) Mover/Unpacker transfers packet back to CPU and
    //       copies results to Grid data structures
    //
    // GPU action B pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU by distributor,
    // 2) GPU action B applied to blocks in packet by Action B GPU team
    // 3) Mover/Unpacker transfers packet back to CPU and
    //    copies results to Grid data structures
    ThreadTeam*        teamA_cpu = teams_[0];
    ThreadTeam*        teamA_gpu = teams_[1];
    ThreadTeam*        teamB_gpu = teams_[2];

    // Assume for no apparent reason that the GPU will finish first
    teamA_gpu->attachThreadReceiver(teamA_cpu);
    teamB_gpu->attachThreadReceiver(teamA_cpu);
    teamA_gpu->attachDataReceiver(&gpuToHost1_);
    teamB_gpu->attachDataReceiver(&gpuToHost2_);

    unsigned int nTotalThreads =       actionA_cpu.nInitialThreads
                                 +     actionA_gpu.nInitialThreads
                                 +     actionB_gpu.nInitialThreads;
    if (nTotalThreads > teamA_cpu->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuGpuWowzaTasks] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    teamA_cpu->startCycle(actionA_cpu, "ActionSharing_CPU_Block_Team");
    teamA_gpu->startCycle(actionA_gpu, "ActionSharing_GPU_Packet_Team");
    teamB_gpu->startCycle(actionB_gpu, "ActionParallel_GPU_Packet_Team");
    gpuToHost1_.startCycle();
    gpuToHost2_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    // Let CPU start work so that we overlap the first host-to-device transfer
    // with CPU computation
    bool        isCpuTurn = true;
    int         nInCpuTurn = 0;

    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    RuntimeBackend&                   backend = RuntimeBackend::instance();
    std::shared_ptr<Tile>             tileA{};
    std::shared_ptr<Tile>             tileB{};
    std::shared_ptr<DataPacket>       packetA_gpu = packetPrototypeA.clone();
    std::shared_ptr<DataPacket>       packetB_gpu = packetPrototypeB.clone();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        tileA = ti->buildCurrentTile();
        tileB = tileA;
        packetB_gpu->addTile( std::move(tileB) );

        // GPU action parallel pipeline
        if (packetB_gpu->nTiles() >= actionB_gpu.nTilesPerPacket) {
            packetB_gpu->pack();
            backend.initiateHostToGpuTransfer(*(packetB_gpu.get()));

            teamB_gpu->enqueue( std::move(packetB_gpu) );
            packetB_gpu = packetPrototypeB.clone();
        }

        // CPU/GPU data parallel pipeline
        if (isCpuTurn) {
            teamA_cpu->enqueue( tilePrototype.clone( std::move(tileA) ) );

            ++nInCpuTurn;
            if (nInCpuTurn >= nTilesPerCpuTurn) {
                isCpuTurn = false;
                nInCpuTurn = 0;
            }
        } else {
            packetA_gpu->addTile( std::move(tileA) );

            if (packetA_gpu->nTiles() >= actionA_gpu.nTilesPerPacket) {
                packetA_gpu->pack();
                backend.initiateHostToGpuTransfer(*(packetA_gpu.get()));

                teamA_gpu->enqueue( std::move(packetA_gpu) );

                packetA_gpu = packetPrototypeA.clone();
                isCpuTurn = true;
            }
        }
    }

    if (packetA_gpu->nTiles() > 0) {
        packetA_gpu->pack();
        backend.initiateHostToGpuTransfer(*(packetA_gpu.get()));
        teamA_gpu->enqueue( std::move(packetA_gpu) );
    } else {
        packetA_gpu.reset();
    }

    if (packetB_gpu->nTiles() > 0) {
        packetB_gpu->pack();
        backend.initiateHostToGpuTransfer(*(packetB_gpu.get()));
        teamB_gpu->enqueue( std::move(packetB_gpu) );
    } else {
        packetB_gpu.reset();
    }

    teamA_cpu->closeQueue(nullptr);
    teamA_gpu->closeQueue(nullptr);
    teamB_gpu->closeQueue(nullptr);

    // We are letting the host thread block without activating a thread in
    // a different thread team.
    teamA_cpu->wait();
    gpuToHost1_.wait();
    gpuToHost2_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    teamA_gpu->detachThreadReceiver();
    teamB_gpu->detachThreadReceiver();
    teamA_gpu->detachDataReceiver();
    teamB_gpu->detachDataReceiver();

    Logger::instance().log("[Runtime] End CPU/GPU shared & GPU configuration");
}
#  endif
#endif

/**
 * 
 *
 * \return 
 */
#ifdef RUNTIME_SUPPORT_DATAPACKETS
#  ifdef RUNTIME_CAN_USE_TILEITER
void Runtime::executeTasks_FullPacket(const std::string& bundleName,
                                      const RuntimeAction& cpuAction,
                                      const RuntimeAction& gpuAction,
                                      const RuntimeAction& postGpuAction,
                                      const DataPacket& packetPrototype) {
    Logger::instance().log("[Runtime] Start CPU/GPU/Post-GPU action bundle");

    if        (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeTasks_FullPacket] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Given GPU action should run on packet of blocks, "
                               "which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeTasks_FullPacket] "
                                    "Need at least one tile per GPU packet");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Given post-GPU action should run on tiles, "
                               "which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeTasks_FullPacket] "
                                    "Post-GPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Need at least three ThreadTeams in runtime");
    }

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    //
    // GPU/Post-GPU action parallel pipeline
    // 1) Asynchronous transfer of Packets of Blocks to GPU
    // 2) GPU action applied to blocks in packet by GPU team
    // 3) Mover/Unpacker transfers packet back to CPU,
    //    copies results to Grid data structures, and
    //    pushes blocks to Post-GPU team
    // 4) Post-GPU action applied by host via Post-GPU team
    ThreadTeam*        cpuTeam     = teams_[0];
    ThreadTeam*        gpuTeam     = teams_[1];
    ThreadTeam*        postGpuTeam = teams_[2];

    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost1_);
    gpuToHost1_.attachDataReceiver(postGpuTeam);

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads
                                 + 1;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                                "Post-GPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    postGpuTeam->startCycle(postGpuAction, "Post_GPU_Block_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    RuntimeBackend&                   backend = RuntimeBackend::instance();
    std::shared_ptr<Tile>             tile_cpu{};
    std::shared_ptr<Tile>             tile_gpu{};
    std::shared_ptr<DataPacket>       packet_gpu = packetPrototype.clone();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        // If we create a first shared_ptr and enqueue it with one team, it is
        // possible that this shared_ptr could have the action applied to its
        // data and go out of scope before we create a second shared_ptr.  In
        // this case, the data item's resources would be released prematurely.
        // To avoid this, we create all copies up front and before enqueing any
        // copy.
        tile_cpu = ti->buildCurrentTile();
        tile_gpu = tile_cpu;
        if ((tile_cpu.get() != tile_gpu.get()) || (tile_cpu.use_count() != 2)) {
            throw std::logic_error("[Runtime::executeTasks_FullPacket] Ownership not shared");
        }

        packet_gpu->addTile( std::move(tile_gpu) );
        if ((tile_gpu != nullptr) || (tile_gpu.use_count() != 0)) {
            throw std::logic_error("[Runtime::executeTasks_FullPacket] tile_gpu ownership not transferred");
        } else if (tile_cpu.use_count() != 2) {
            throw std::logic_error("[Runtime::executeTasks_FullPacket] Ownership not shared after transfer");
        }

        // CPU action parallel pipeline
        cpuTeam->enqueue( std::move(tile_cpu) );
        if ((tile_cpu != nullptr) || (tile_cpu.use_count() != 0)) {
            throw std::logic_error("[Runtime::executeTasks_FullPacket] tile_cpu ownership not transferred");
        }

        // GPU/Post-GPU action parallel pipeline
        if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
            packet_gpu->pack();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));

            gpuTeam->enqueue( std::move(packet_gpu) );
            if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
                throw std::logic_error("[Runtime::executeTasks_FullPacket] packet_gpu ownership not transferred");
            }

            packet_gpu = packetPrototype.clone();
        }
    }

    if (packet_gpu->nTiles() > 0) {
        packet_gpu->pack();
        backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
        gpuTeam->enqueue( std::move(packet_gpu) );
    } else {
        packet_gpu.reset();
    }

    if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] packet_gpu ownership not transferred (after)");
    }

    gpuTeam->closeQueue(nullptr);
    cpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread 
    // in the host team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();
    postGpuTeam->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    cpuTeam->detachThreadReceiver();
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();
    gpuToHost1_.detachDataReceiver();

    Logger::instance().log("[Runtime] End CPU/GPU/Post-GPU action bundle");
}
#  endif
#endif

}

