#include "Runtime.h"

#ifdef USE_THREADED_DISTRIBUTOR
#include <omp.h>
#include <cstdio>
#endif

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "Grid.h"
#include "DataPacket.h"
#include "OrchestrationLogger.h"
#include "StreamManager.h"

#ifdef USE_CUDA_BACKEND
// TODO: Should these be designed like StreamManager so that the Runtime class
// is not dealing with backend-specific details?
#include "CudaGpuEnvironment.h"
#include "CudaMemoryManager.h"
#endif

#include "Flash.h"

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

    // Create/initialize singletons needed by runtime
    orchestration::StreamManager::instantiate(nStreams);
#ifdef USE_CUDA_BACKEND
    orchestration::CudaMemoryManager::instantiate(nBytesInMemoryPools);
#endif

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

#ifdef USE_CUDA_BACKEND
    // TODO: Should this class have an instantiate message and log this
    // message itself when instantiated?
    CudaGpuEnvironment&    gpuEnv = CudaGpuEnvironment::instance();
    std::string   msg =   "[Runtime] " 
                        + std::to_string(gpuEnv.nGpuDevices()) 
                        + " GPU device(s) per process found\n"
                        + gpuEnv.information();
    Logger::instance().log(msg);
#endif
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
void Runtime::executeTasks(const ActionBundle& bundle,
                           const unsigned int nDistributorThreads) {
    std::string msg = "[Runtime] Start execution of " + bundle.name;
    Logger::instance().log(msg);

    bool  hasCpuAction     = (bundle.cpuAction.routine     != nullptr);
    bool  hasGpuAction     = (bundle.gpuAction.routine     != nullptr);
    bool  hasPostGpuAction = (bundle.postGpuAction.routine != nullptr);

    // TODO: The pipeline construction would be done dynamically.
    // Realistically, we would have multiple different implementations and 
    // this routine would select the setup based on the parameter values.
    // The assignment of team type of team ID would be hardcoded in each.
    //
    // TASK_COMPOSER: Should the task composer identify the pipelines that it
    // will need and then write this routine for each?  If not, the
    // combinatorics could grow out of control fairly quickly.
    if (hasCpuAction && !hasGpuAction && !hasPostGpuAction) {
        executeCpuTasks(bundle.name, nDistributorThreads, bundle.cpuAction);
#if defined(USE_CUDA_BACKEND)
    } else if (!hasCpuAction && hasGpuAction && !hasPostGpuAction) {
        executeGpuTasks(bundle.name,
                        bundle.gpuAction);
    } else if (   hasCpuAction && hasGpuAction && hasPostGpuAction
               && (bundle.cpuAction.teamType     == ThreadTeamDataType::BLOCK)
               && (bundle.gpuAction.teamType     == ThreadTeamDataType::SET_OF_BLOCKS)
               && (bundle.postGpuAction.teamType == ThreadTeamDataType::BLOCK)) {
        executeTasks_FullPacket(bundle.name,
                                bundle.cpuAction,
                                bundle.gpuAction,
                                bundle.postGpuAction);
#endif
    } else {
        std::string   errMsg =   "[Runtime::executeTasks] ";
        errMsg += "No compatible thread team layout - ";
        errMsg += bundle.name;
        errMsg += "\n";
        throw std::logic_error(errMsg);
    }

    msg  = "[Runtime] Finished execution of " + bundle.name;
    Logger::instance().log(msg);
}

/**
 * 
 *
 * \return 
 */
void Runtime::executeCpuTasks(const std::string& actionName,
                              const unsigned int nDistributorThreads,
                              const RuntimeAction& cpuAction) {
    Logger::instance().log("[Runtime] Start single CPU action");

    if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeCpuTasks] "
                               "Given CPU action should run on tiles, "
                               "which is not in configuration");
    } else if (nDistributorThreads <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuTasks] "
                                    "nDistributorThreads must be positive");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime::executeCpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif

    //***** ASSEMBLE THREAD TEAM CONFIGURATION
    // CPU action parallel pipeline
    // 1) CPU action applied to blocks by CPU team
    ThreadTeam*   cpuTeam = teams_[0];

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads = cpuAction.nInitialThreads + nDistThreads;
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
    int                       tId{-1};
    std::shared_ptr<Tile>     tileDesc{};
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, level, cpuTeam) \
                     private(tId, tileDesc) \
                     num_threads(nDistThreads)
#endif
    {
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            tileDesc = ti->buildCurrentTile();
//#ifdef USE_THREADED_DISTRIBUTOR
//            tId = omp_get_thread_num();
//            printf("[Thread %d] Working on block %d\n", tId, tileDesc->gridIndex());
//#endif
            cpuTeam->enqueue(std::move(tileDesc));
//            cpuTeam->enqueue( ti->buildCurrentTile() );
        }
        cpuTeam->increaseThreadCount(1);
    }
    cpuTeam->closeQueue();
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
                              const RuntimeAction& gpuAction) {
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

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                  level = 0;
    Grid&                         grid = Grid::instance();
    std::shared_ptr<DataPacket>   packet_gpu = DataPacket::createPacket();
    if ((packet_gpu == nullptr) || (packet_gpu.use_count() != 1)) {
        throw std::logic_error("[Runtime::executeGpuTasks] Bad packet at creation");
    }
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        packet_gpu->addTile( ti->buildCurrentTile() );
        if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
            packet_gpu->initiateHostToDeviceTransfer();

            gpuTeam->enqueue( std::move(packet_gpu) );
            if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
                throw std::logic_error("[Runtime::executeGpuTasks] Ownership not transferred (in loop)");
            }

            packet_gpu = DataPacket::createPacket();
        }
    }

    if (packet_gpu->nTiles() > 0) {
        packet_gpu->initiateHostToDeviceTransfer();
        gpuTeam->enqueue( std::move(packet_gpu) );
    } else {
        packet_gpu.reset();
    }
    if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
        throw std::logic_error("[Runtime::executeGpuTasks] Ownership not transferred (after)");
    }

    gpuTeam->closeQueue();

    // host thread blocks until cycle ends, so activate another thread 
    // in team first
    gpuTeam->increaseThreadCount(1);
    gpuTeam->wait();

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
                                 const RuntimeAction& gpuAction) {
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

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    std::shared_ptr<Tile>             tile_cpu{};
    std::shared_ptr<Tile>             tile_gpu{};
    std::shared_ptr<DataPacket>       packet_gpu = DataPacket::createPacket();
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
            packet_gpu->initiateHostToDeviceTransfer();

            gpuTeam->enqueue( std::move(packet_gpu) );
            if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
                throw std::logic_error("[Runtime::executeCpuGpuTasks] packet_gpu ownership not transferred");
            }

            packet_gpu = DataPacket::createPacket();
        }
    }

    if (packet_gpu->nTiles() > 0) {
        packet_gpu->initiateHostToDeviceTransfer();
        gpuTeam->enqueue( std::move(packet_gpu) );
    } else {
        packet_gpu.reset();
    }

    if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] packet_gpu ownership not transferred (after)");
    }

    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    // host thread blocks until cycle ends, so activate another thread 
    // in the host team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();
    gpuTeam->wait();

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
                                      const RuntimeAction& postGpuAction) {
#ifdef USE_THREADED_DISTRIBUTOR
    const unsigned int  nDistThreads = nDistributorThreads;
#else
    const unsigned int  nDistThreads = 1;
#endif
    std::string   msg =   "[Runtime] Start GPU/Post-GPU action bundle - "
                        + std::to_string(nDistThreads)
                        + " Distributor Threads";
    Logger::instance().log(msg);

    if        (nDistributorThreads <= 0) {
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

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
#ifdef USE_THREADED_DISTRIBUTOR
#pragma omp parallel default(none) \
                     shared(grid, level, gpuTeam, gpuAction, postGpuTeam) \
                     num_threads(nDistThreads)
#endif
    {
        std::shared_ptr<DataPacket>       packet_gpu = DataPacket::createPacket();
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            packet_gpu->addTile( ti->buildCurrentTile() );

            if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
                packet_gpu->initiateHostToDeviceTransfer();

                gpuTeam->enqueue( std::move(packet_gpu) );

                packet_gpu = DataPacket::createPacket();
            }
        }

        if (packet_gpu->nTiles() > 0) {
            packet_gpu->initiateHostToDeviceTransfer();
            gpuTeam->enqueue( std::move(packet_gpu) );
        } else {
            packet_gpu.reset();
        }

        // host thread blocks until cycle ends, so activate a thread
        postGpuTeam->increaseThreadCount(1);
    }

    gpuTeam->closeQueue();
    gpuTeam->wait();
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
                                      const RuntimeAction& cpuAction,
                                      const RuntimeAction& gpuAction,
                                      const unsigned int nTilesPerCpuTurn) {
    Logger::instance().log("[Runtime] Start CPU/GPU shared action");
    std::string   msg = "[Runtime] "
                        + std::to_string(nTilesPerCpuTurn)
                        + " tiles sent to CPU for every packet of "
                        + std::to_string(gpuAction.nTilesPerPacket)
                        + " tiles sent to GPU";
    Logger::instance().log(msg);

    if        (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
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

    // Assume for no apparent reason that the GPU will finish first
    gpuTeam->attachThreadReceiver(cpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost1_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 +     1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                                "CPU could receive too many thread "
                                "activation calls");
    }

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "ActionSharing_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "ActionSharing_GPU_Packet_Team");

    //***** ACTION PARALLEL DISTRIBUTOR
    // Let CPU start work so that we overlap the first host-to-device transfer
    // with CPU computation
    bool        isCpuTurn = true;
    int         nInCpuTurn = 0;

    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    std::shared_ptr<Tile>             tileDesc{};
    std::shared_ptr<DataPacket>       packet_gpu = DataPacket::createPacket();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        tileDesc = ti->buildCurrentTile();
        if ((tileDesc == nullptr) || (tileDesc.use_count() != 1)) {
            throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] tileDesc acquisition failed");
        }

        if (isCpuTurn) {
            cpuTeam->enqueue( std::move(tileDesc) );
            if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
                throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] tileDesc ownership not transferred/CPU");
            }

            ++nInCpuTurn;
            if (nInCpuTurn >= nTilesPerCpuTurn) {
                isCpuTurn = false;
                nInCpuTurn = 0;
            }
        } else {
            packet_gpu->addTile( std::move(tileDesc) );
            if ((tileDesc != nullptr) || (tileDesc.use_count() != 0)) {
                throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] tileDesc ownership not transferred/GPU");
            }

            if (packet_gpu->nTiles() >= gpuAction.nTilesPerPacket) {
                packet_gpu->initiateHostToDeviceTransfer();

                gpuTeam->enqueue( std::move(packet_gpu) );
                if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
                    throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] packet_gpu ownership not transferred");
                }

                packet_gpu = DataPacket::createPacket();
                isCpuTurn = true;
            }
        }
    }

    if (packet_gpu->nTiles() > 0) {
        packet_gpu->initiateHostToDeviceTransfer();
        gpuTeam->enqueue( std::move(packet_gpu) );
    } else {
        packet_gpu.reset();
    }

    if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] packet_gpu ownership not transferred (after)");
    }

    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    // host thread blocks until cycle ends, so activate another thread 
    // in the device team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();
    gpuTeam->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachThreadReceiver();
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
void Runtime::executeCpuGpuWowzaTasks(const std::string& bundleName,
                                      const RuntimeAction& actionA_cpu,
                                      const RuntimeAction& actionA_gpu,
                                      const RuntimeAction& actionB_gpu,
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

    //***** ACTION PARALLEL DISTRIBUTOR
    // Let CPU start work so that we overlap the first host-to-device transfer
    // with CPU computation
    bool        isCpuTurn = true;
    int         nInCpuTurn = 0;

    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    std::shared_ptr<Tile>             tileA{};
    std::shared_ptr<Tile>             tileB{};
    std::shared_ptr<DataPacket>       packetA_gpu = DataPacket::createPacket();
    std::shared_ptr<DataPacket>       packetB_gpu = DataPacket::createPacket();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        tileA = ti->buildCurrentTile();
        tileB = tileA;
        packetB_gpu->addTile( std::move(tileB) );

        // GPU action parallel pipeline
        if (packetB_gpu->nTiles() >= actionB_gpu.nTilesPerPacket) {
            packetB_gpu->initiateHostToDeviceTransfer();

            teamB_gpu->enqueue( std::move(packetB_gpu) );
            packetB_gpu = DataPacket::createPacket();
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
                packetA_gpu->initiateHostToDeviceTransfer();

                teamA_gpu->enqueue( std::move(packetA_gpu) );

                packetA_gpu = DataPacket::createPacket();
                isCpuTurn = true;
            }
        }
    }

    if (packetA_gpu->nTiles() > 0) {
        packetA_gpu->initiateHostToDeviceTransfer();
        teamA_gpu->enqueue( std::move(packetA_gpu) );
    } else {
        packetA_gpu.reset();
    }

    if (packetB_gpu->nTiles() > 0) {
        packetB_gpu->initiateHostToDeviceTransfer();
        teamB_gpu->enqueue( std::move(packetB_gpu) );
    } else {
        packetB_gpu.reset();
    }

    teamA_cpu->closeQueue();
    teamA_gpu->closeQueue();
    teamB_gpu->closeQueue();

    // We are letting the host thread block without activating a thread in
    // a different thread team.
    teamA_cpu->wait();
    teamA_gpu->wait();
    teamB_gpu->wait();

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
                                      const RuntimeAction& postGpuAction) {
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

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int                      level = 0;
    Grid&                             grid = Grid::instance();
    std::shared_ptr<Tile>             tile_cpu{};
    std::shared_ptr<Tile>             tile_gpu{};
    std::shared_ptr<DataPacket>       packet_gpu = DataPacket::createPacket();
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
            packet_gpu->initiateHostToDeviceTransfer();

            gpuTeam->enqueue( std::move(packet_gpu) );
            if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
                throw std::logic_error("[Runtime::executeTasks_FullPacket] packet_gpu ownership not transferred");
            }

            packet_gpu = DataPacket::createPacket();
        }
    }

    if (packet_gpu->nTiles() > 0) {
        packet_gpu->initiateHostToDeviceTransfer();
        gpuTeam->enqueue( std::move(packet_gpu) );
    } else {
        packet_gpu.reset();
    }

    if ((packet_gpu != nullptr) || (packet_gpu.use_count() != 0)) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] packet_gpu ownership not transferred (after)");
    }

    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    // host thread blocks until cycle ends, so activate another thread 
    // in the host team first
    cpuTeam->increaseThreadCount(1);
    cpuTeam->wait();
    gpuTeam->wait();
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

