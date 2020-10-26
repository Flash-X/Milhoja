// WIP: Somehow NDEBUG is getting set and deactivating the asserts
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include "Runtime.h"

#include <stdexcept>
#include <iostream>

#include "ThreadTeam.h"
#include "Grid.h"
#include "DataPacket.h"
#include "OrchestrationLogger.h"

#ifdef USE_CUDA_BACKEND
#include "CudaGpuEnvironment.h"
#include "CudaStreamManager.h"
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
Runtime& Runtime::instance(void) {
    static Runtime     orSingleton;
    return orSingleton;
}

/**
 * 
 *
 * \return 
 */
void Runtime::setLogFilename(const std::string& filename) {
    orchestration::Logger::setLogFilename(filename);
}

/**
 * 
 *
 * \return 
 */
void Runtime::setNumberThreadTeams(const unsigned int nTeams) {
    if (instantiated_) {
        throw std::logic_error("[Runtime::setNumberThreadTeams] "
                               "Set only when runtime does not exist");
    } else if (nTeams == 0) {
        throw std::invalid_argument("[Runtime::setNumberThreadTeams] "
                                    "Need at least one ThreadTeam");
    }

    nTeams_ = nTeams;
}

/**
 * 
 *
 * \return 
 */
void Runtime::setMaxThreadsPerTeam(const unsigned int nThreads) {
    if (instantiated_) {
        throw std::logic_error("[Runtime::setMaxThreadsPerTeam] "
                               "Set only when runtime does not exist");
    } else if (nThreads == 0) {
        throw std::invalid_argument("[Runtime::setMaxThreadsPerTeam] "
                                    "Need at least one thread per team");
    }

    maxThreadsPerTeam_ = nThreads;
}

/**
 * 
 *
 * \return 
 */
Runtime::Runtime(void)
    : teams_{nullptr}
{
    Logger::instance().log("[Runtime] Initializing...");

    if (nTeams_ <= 0) {
        throw std::invalid_argument("[Runtime::Runtime] "
                                    "Need to create at least one team");
    }

    teams_ = new ThreadTeam*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam(maxThreadsPerTeam_, i);
    }

#ifdef USE_CUDA_BACKEND
    CudaGpuEnvironment&    gpuEnv = CudaGpuEnvironment::instance();
    std::string   msg =   "[Runtime] " 
                        + std::to_string(gpuEnv.nGpuDevices()) 
                        + " GPU device(s) per process found\n"
                        + gpuEnv.information();
    Logger::instance().log(msg);

    CudaStreamManager::instance();
    CudaMemoryManager::instance();
#endif

    instantiated_ = true;

    Logger::instance().log("[Runtime] Created and ready for use");
}

/**
 * 
 *
 * \return 
 */
Runtime::~Runtime(void) {
    Logger::instance().log("[Runtime] Finalizing...");

    instantiated_ = false;

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }
    delete [] teams_;
    teams_ = nullptr;

    Logger::instance().log("[Runtime] Finalized");
}

/**
 * 
 *
 * \return 
 */
void Runtime::executeTasks(const ActionBundle& bundle) {
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
        executeCpuTasks(bundle.name,
                        bundle.cpuAction);
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
                              const RuntimeAction& cpuAction) {
    Logger::instance().log("[Runtime] Start single CPU action");

    if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeCpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
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

    //***** START EXECUTION CYCLE
    cpuTeam->startCycle(cpuAction, "CPU_Block_Team");

    //***** ACTION PARALLEL DISTRIBUTOR
    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        cpuTeam->enqueue( ti->buildCurrentTile() );
    }
    cpuTeam->closeQueue();

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
                              const RuntimeAction& gpuAction) {
    Logger::instance().log("[Runtime] Start single GPU action");

    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeGpuTasks] "
                               "Given GPU action should run on a thread team "
                               "that works with data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeGpuTasks] "
                                    "Need at least one tile per packet");
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
    gpuTeam->attachDataReceiver(&gpuToHost_);

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
                               "Given CPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] "
                               "Given GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuTasks] "
                                    "Need at least one tile per GPU packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] "
                               "Need at least three ThreadTeams in runtime");
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
    gpuTeam->attachDataReceiver(&gpuToHost_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 +     1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuGpuTasks] "
                                "CPU could receive too many thread "
                                "activation calls from GPU team");
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

        // GPU/Post-GPU action parallel pipeline
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
                               "Given CPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                               "Given GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeCpuGpuSplitTasks] "
                                    "Need at least one tile per GPU packet");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                               "Need at least three ThreadTeams in runtime");
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
    gpuTeam->attachDataReceiver(&gpuToHost_);

    // The action parallel distributor's thread resource is used
    // once the distributor starts to wait
    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 +     1;
    if (nTotalThreads > cpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] "
                                "CPU could receive too many thread "
                                "activation calls from GPU team");
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
            throw std::logic_error("[Runtime::executeCpuGpuSplitTasks] tileDesc ownership not transferred");
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
void Runtime::executeTasks_FullPacket(const std::string& bundleName,
                                      const RuntimeAction& cpuAction,
                                      const RuntimeAction& gpuAction,
                                      const RuntimeAction& postGpuAction) {
    Logger::instance().log("[Runtime] Start CPU/GPU/Post-GPU action bundle");

    if        (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Given CPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeTasks_FullPacket] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Given GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[Runtime::executeTasks_FullPacket] "
                                    "Need at least one tile per GPU packet");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Given post-GPU action should run on tile-based "
                               "thread team, which is not in configuration");
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
    gpuTeam->attachDataReceiver(&gpuToHost_);
    gpuToHost_.attachDataReceiver(postGpuTeam);

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads
                                 + 1;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                                "Post-GPU could receive too many thread "
                                "activation calls from CPU and GPU teams");
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
    gpuToHost_.detachDataReceiver();

    Logger::instance().log("[Runtime] End CPU/GPU/Post-GPU action bundle");
}
#endif

}

