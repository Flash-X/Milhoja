#include "Runtime.h"

#include <mpi.h>
#ifdef USE_THREADED_DISTRIBUTOR
#include <omp.h>
#endif

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "Grid.h"
#include "Backend.h"
#include "DataPacket.h"
#include "OrchestrationLogger.h"

#include "Flash.h"
#include "constants.h"

namespace orchestration {

unsigned int    Runtime::nTeams_            = 0;
unsigned int    Runtime::maxThreadsPerTeam_ = 0;
bool            Runtime::instantiated_      = false;

/**
 * 
 *
 * \return 
 */
void   Runtime::instantiate(const unsigned int nTeams,
                            const unsigned int nThreadsPerTeam,
                            const unsigned int nStreams,
                            const std::size_t  nBytesInMemoryPools) {
    Logger::instance().log("[Runtime] Initializing...");

    if (instantiated_) {
        throw std::logic_error("[Runtime::instantiate] Already instantiated");
    } else if (nTeams == 0) {
        throw std::invalid_argument("[Runtime::instantiate] "
                                    "Need at least one ThreadTeam");
    } else if (nThreadsPerTeam == 0) {
        throw std::invalid_argument("[Runtime::instantiate] "
                                    "Need at least one thread per team");
    }

    nTeams_ = nTeams;
    maxThreadsPerTeam_ = nThreadsPerTeam;
    instantiated_ = true;

    orchestration::Backend::instantiate(nStreams, nBytesInMemoryPools);

    // Create/initialize runtime
    instance();

    Logger::instance().log("[Runtime] Created and ready for use");
}

/**
 * 
 *
 * \return 
 */
Runtime& Runtime::instance(void) {
    if (!instantiated_) {
        throw std::logic_error("[Runtime::instance] Instantiate first");
    }

    static Runtime     singleton;
    return singleton;
}

/**
 * 
 *
 * \return 
 */
Runtime::Runtime(void)
    : teams_{nullptr}
{
    teams_ = new ThreadTeam*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam(maxThreadsPerTeam_, i);
    }
}

/**
 * 
 *
 * \return 
 */
Runtime::~Runtime(void) {
    Logger::instance().log("[Runtime] Finalizing...");

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }
    delete [] teams_;
    teams_ = nullptr;

    instantiated_ = false;

    Logger::instance().log("[Runtime] Finalized");
}

/**
 * 
 *
 * \return 
 */
void Runtime::executeCpuTasks(const std::string& actionName,
                              const RuntimeAction& cpuAction) {
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
    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        cpuTeam->enqueue( ti->buildCurrentTile() );
    }
    cpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread 
    // in team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();

    // No need to break apart the thread team configuration

    Logger::instance().log("[Runtime] End single CPU action");
}

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void Runtime::executeGpuTasks(const std::string& bundleName,
                              const RuntimeAction& gpuAction,
                              const DataPacket& packetPrototype,
                              const unsigned int stepNumber) {
    Logger::instance().log("[Runtime] Start single GPU action");

    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
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

    Grid&         grid = Grid::instance();
    Backend&      backend = Backend::instance();

    //***** SETUP TIMING
    int rank = -1;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    unsigned int  nPackets = ceil(  (double)grid.getNumberLocalBlocks()
                                  / (double)gpuAction.nTilesPerPacket);

    unsigned int  pIdx         = 0;
    double        tStartPacket = 0.0;
    double        tStartPack   = 0.0;
    double        tStartAsync  = 0.0;
    unsigned int  bCounts[nPackets];
    double        wtimesPack_sec[nPackets];
    double        wtimesAsync_sec[nPackets];
    double        wtimesPacket_sec[nPackets];

    std::string   filename("timings_packet_step");
    filename += std::to_string(stepNumber);
    filename += "_rank";
    filename += std::to_string(rank);
    filename += ".dat";

    std::ofstream   fptr;
    fptr.open(filename, std::ios::out);
    fptr << "# Testname = GPU-Only\n";
    fptr << "# Dimension = " << NDIM << "\n";
    fptr << "# NXB = " << NXB << "\n";
    fptr << "# NYB = " << NYB << "\n";
    fptr << "# NZB = " << NZB << "\n";
    fptr << "# n_blocks_per_packet = " << gpuAction.nTilesPerPacket << "\n";
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

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads = gpuAction.nInitialThreads + 1;
    if (nTotalThreads > gpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeGpuTasks] "
                               "GPU team could receive too many thread "
                               "activation calls from distributor");
    }

    //***** START EXECUTION CYCLE
    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int  level = 0;

    tStartPacket = MPI_Wtime();
    std::shared_ptr<DataPacket>   packet_gpu = packetPrototype.clone();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        packet_gpu->addTile( ti->buildCurrentTile() );
        if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
            tStartPack = MPI_Wtime();
            packet_gpu->pack();
            wtimesPack_sec[pIdx] = MPI_Wtime() - tStartPack;

            tStartAsync = MPI_Wtime();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            wtimesAsync_sec[pIdx] = MPI_Wtime() - tStartAsync;

            bCounts[pIdx] = packet_gpu->nTiles();
            gpuTeam->enqueue( std::move(packet_gpu) );

            wtimesPacket_sec[pIdx] = MPI_Wtime() - tStartPacket;
            ++pIdx;

            tStartPacket = MPI_Wtime();
            packet_gpu = packetPrototype.clone();
        }
    }

    if (packet_gpu->nTiles() > 0) {
        tStartPack = MPI_Wtime();
        packet_gpu->pack();
        wtimesPack_sec[pIdx] = MPI_Wtime() - tStartPack;

        tStartAsync = MPI_Wtime();
        backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
        wtimesAsync_sec[pIdx] = MPI_Wtime() - tStartAsync;

        bCounts[pIdx] = packet_gpu->nTiles();
        gpuTeam->enqueue( std::move(packet_gpu) );

        wtimesPacket_sec[pIdx] = MPI_Wtime() - tStartPacket;
    } else {
        packet_gpu.reset();
    }

    gpuTeam->closeQueue(nullptr);

    // host thread blocks until cycle ends, so activate another thread 
    // in team first
    gpuTeam->increaseThreadCount(1);

    fptr << std::setprecision(15);
    for (pIdx=0; pIdx<nPackets; ++pIdx) {
        fptr << '0,' << pIdx << ','
             << bCounts[pIdx] << ','
             << wtimesPack_sec[pIdx] << ','
             << wtimesAsync_sec[pIdx] << ','
             << wtimesPacket_sec[pIdx] << "\n";
    }
    fptr.close();

    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime] End single GPU action");
}
#endif

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void Runtime::executeCpuGpuTasks(const std::string& bundleName,
                                 const RuntimeAction& cpuAction,
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
    Backend&                          backend = Backend::instance();
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
        cpuTeam->enqueue( std::move(tile_cpu) );
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
#endif

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void Runtime::executeExtendedGpuTasks(const std::string& bundleName,
                                      const unsigned int nDistributorThreads,
                                      const RuntimeAction& gpuAction,
                                      const RuntimeAction& postGpuAction,
                                      const DataPacket& packetPrototype) {
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
    Backend&                          backend = Backend::instance();
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
#endif

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void Runtime::executeCpuGpuSplitTasks(const std::string& bundleName,
                                      const unsigned int nDistributorThreads,
                                      const RuntimeAction& cpuAction,
                                      const RuntimeAction& gpuAction,
                                      const DataPacket& packetPrototype,
                                      const unsigned int nTilesPerCpuTurn,
                                      const unsigned int stepNumber) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif
    Logger::instance().log("[Runtime] Start CPU/GPU shared action");
    std::string   msg =   "[Runtime] "
                        + std::to_string(nDistThreads)
                        + " distributor threads";
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

    Grid&      grid = Grid::instance();
    Backend&   backend = Backend::instance();

    //***** SETUP TIMING
    int rank = -1;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    unsigned int  nPackets = ceil(  (double)grid.getNumberLocalBlocks()
                                  / (double)gpuAction.nTilesPerPacket);

    unsigned int  pCounts[nDistThreads];
    unsigned int  bCounts[nPackets][nDistThreads];
    double        wtimesPack_sec[nPackets][nDistThreads];
    double        wtimesAsync_sec[nPackets][nDistThreads];
    double        wtimesPacket_sec[nPackets][nDistThreads];

    std::string   filename("timings_packet_step");
    filename += std::to_string(stepNumber);
    filename += "_rank";
    filename += std::to_string(rank);
    filename += ".dat";

    std::ofstream   fptr;
    fptr.open(filename, std::ios::out);
    fptr << "# Testname = Data Parallel CPU/GPU\n";
    fptr << "# Dimension = " << NDIM << "\n";
    fptr << "# NXB = " << NXB << "\n";
    fptr << "# NYB = " << NYB << "\n";
    fptr << "# NZB = " << NZB << "\n";
    fptr << "# n_blocks_per_packet = " << gpuAction.nTilesPerPacket << "\n";
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
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "ActionSharing_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "ActionSharing_GPU_Packet_Team");
    gpuToHost1_.startCycle();

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int   level = 0;
    // TODO:  A first look at this with NSight makes it look like the OMP
    // threads are busy-waiting (aka spin cycling) at the implied barrier.
    // Investigate this.
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, backend, level, packetPrototype, \
                            cpuTeam, gpuTeam, gpuAction, nTilesPerCpuTurn, \
                            rank, \
                            wtimesPack_sec, wtimesAsync_sec, wtimesPacket_sec, \
                            pCounts, bCounts) \
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

        unsigned int  pIdx         = 0;
        double        tStartPacket = 0.0;
        double        tStartPack   = 0.0;
        double        tStartAsync  = 0.0;

        pCounts[tId] = 0;

        std::shared_ptr<Tile>             tileDesc{};

        tStartPacket = MPI_Wtime();
        std::shared_ptr<DataPacket>       packet_gpu = packetPrototype.clone();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();

            if (isCpuTurn) {
                cpuTeam->enqueue( std::move(tileDesc) );

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
                    wtimesPack_sec[pIdx][tId] = MPI_Wtime() - tStartPack;

                    tStartAsync = MPI_Wtime();
                    backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
                    wtimesAsync_sec[pIdx][tId] = MPI_Wtime() - tStartAsync;

                    bCounts[pIdx][tId] = packet_gpu->nTiles();
                    gpuTeam->enqueue( std::move(packet_gpu) );

                    wtimesPacket_sec[pIdx][tId] = MPI_Wtime() - tStartPacket;

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
            wtimesPack_sec[pIdx][tId] = MPI_Wtime() - tStartPack;

            tStartAsync = MPI_Wtime();
            backend.initiateHostToGpuTransfer(*(packet_gpu.get()));
            wtimesAsync_sec[pIdx][tId] = MPI_Wtime() - tStartAsync;

            bCounts[pIdx][tId] = packet_gpu->nTiles();
            gpuTeam->enqueue( std::move(packet_gpu) );

            wtimesPacket_sec[pIdx][tId] = MPI_Wtime() - tStartPacket;

            ++pIdx;
        } else {
            packet_gpu.reset();
        }
        pCounts[tId] = pIdx;

        cpuTeam->increaseThreadCount(1);
    } // implied barrier
    gpuTeam->closeQueue(nullptr);
    cpuTeam->closeQueue(nullptr);

    fptr << std::setprecision(15);
    for     (unsigned int tIdx=0; tIdx<nDistThreads;  ++tIdx) {
        for (unsigned int pIdx=0; pIdx<pCounts[tIdx]; ++pIdx) {
            fptr << tIdx << ',' << pIdx << ','
                 << bCounts[pIdx][tIdx] << ','
                 << wtimesPack_sec[pIdx][tIdx] << ','
                 << wtimesAsync_sec[pIdx][tIdx] << ','
                 << wtimesPacket_sec[pIdx][tIdx] << "\n";
        }
    }
    fptr.close();

    cpuTeam->wait();
    gpuToHost1_.wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[Runtime] End CPU/GPU shared action");
}
#endif

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void Runtime::executeExtendedCpuGpuSplitTasks(const std::string& bundleName,
                                              const unsigned int nDistributorThreads,
                                              const RuntimeAction& actionA_cpu,
                                              const RuntimeAction& actionA_gpu,
                                              const RuntimeAction& postActionB_cpu,
                                              const DataPacket& packetPrototype,
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
    teamA_gpu->attachDataReceiver(&gpuToHost1_);
    gpuToHost1_.attachDataReceiver(teamB_cpu);

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
    unsigned int  level = 0;
    Grid&         grid = Grid::instance();
    Backend&      backend = Backend::instance();
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
                teamA_cpu->enqueue( std::move(tileDesc) );

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
#endif

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void Runtime::executeCpuGpuWowzaTasks(const std::string& bundleName,
                                      const RuntimeAction& actionA_cpu,
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
    Backend&                          backend = Backend::instance();
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
            teamA_cpu->enqueue( std::move(tileA) );

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
#endif

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
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
    Backend&                          backend = Backend::instance();
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
#endif

}

