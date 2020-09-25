#include "CudaRuntime.h"

#include <stdexcept>
#include <iostream>

#include "ThreadTeam.h"
#include "Grid.h"
#include "OrchestrationLogger.h"

#ifdef USE_CUDA_BACKEND
#include "CudaGpuEnvironment.h"
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"
#include "CudaDataPacket.h"
#endif

#include "Flash.h"

namespace orchestration {

unsigned int    CudaRuntime::nTeams_            = 0;
unsigned int    CudaRuntime::maxThreadsPerTeam_ = 0;
bool            CudaRuntime::instantiated_      = false;

/**
 * 
 *
 * \return 
 */
CudaRuntime& CudaRuntime::instance(void) {
    static CudaRuntime     orSingleton;
    return orSingleton;
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::setLogFilename(const std::string& filename) {
    orchestration::Logger::setLogFilename(filename);
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::setNumberThreadTeams(const unsigned int nTeams) {
    if (instantiated_) {
        throw std::logic_error("[CudaRuntime::setNumberThreadTeams] "
                               "Set only when runtime does not exist");
    } else if (nTeams == 0) {
        throw std::invalid_argument("[CudaRuntime::setNumberThreadTeams] "
                                    "Need at least one ThreadTeam");
    }

    nTeams_ = nTeams;
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::setMaxThreadsPerTeam(const unsigned int nThreads) {
    if (instantiated_) {
        throw std::logic_error("[CudaRuntime::setMaxThreadsPerTeam] "
                               "Set only when runtime does not exist");
    } else if (nThreads == 0) {
        throw std::invalid_argument("[CudaRuntime::setMaxThreadsPerTeam] "
                                    "Need at least one thread per team");
    }

    maxThreadsPerTeam_ = nThreads;
}

/**
 * 
 *
 * \return 
 */
CudaRuntime::CudaRuntime(void)
    : teams_{nullptr}
{
    Logger::instance().log("[CudaRuntime] Initializing...");

    if (nTeams_ <= 0) {
        throw std::invalid_argument("[CudaRuntime::CudaRuntime] "
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

    Logger::instance().log("[CudaRuntime] Created and ready for use");
}

/**
 * 
 *
 * \return 
 */
CudaRuntime::~CudaRuntime(void) {
    Logger::instance().log("[CudaRuntime] Finalizing...");

    instantiated_ = false;

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }
    delete [] teams_;
    teams_ = nullptr;

    Logger::instance().log("[CudaRuntime] Finalized");
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::executeCpuTasks(const std::string& actionName,
                                  const RuntimeAction& cpuAction) {
    Logger::instance().log("[Runtime] Start single CPU action");

    if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[CudaRuntime::executeCpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[CudaRuntime::executeCpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[CudaRuntime::executeCpuTasks] "
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

    // host thread blocks until cycle ends
    cpuTeam->wait();

    // No need to break apart the thread team configuration

    Logger::instance().log("[CudaRuntime] End single CPU action");
}

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void CudaRuntime::executeGpuTasks(const std::string& bundleName,
                                  const RuntimeAction& gpuAction) {
    Logger::instance().log("[Runtime] Start single GPU action");

    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[CudaRuntime::executeGpuTasks] "
                               "Given GPU action should run on a thread team "
                               "that works with data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[CudaRuntime::executeGpuTasks] "
                                    "Need at least one tile per packet");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[CudaRuntime::executeGpuTasks] "
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

    unsigned int   level = 0;
    Grid&          grid = Grid::instance();
    auto           packet_gpu = std::shared_ptr<DataPacket>{};
    assert(packet_gpu == nullptr);
    assert(packet_gpu.use_count() == 0);
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        packet_gpu = std::shared_ptr<DataPacket>{ new CudaDataPacket{ ti->buildCurrentTile() } };
        packet_gpu->initiateHostToDeviceTransfer();

        gpuTeam->enqueue( std::move(packet_gpu) );
        assert(packet_gpu == nullptr);
        assert(packet_gpu.use_count() == 0);
    }

    gpuTeam->closeQueue();

    // host thread blocks until cycle ends
    gpuTeam->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    gpuTeam->detachDataReceiver();

    Logger::instance().log("[CudaRuntime] End single GPU action");
}
#endif

/**
 * 
 *
 * \return 
 */
#if defined(USE_CUDA_BACKEND)
void CudaRuntime::executeTasks_FullPacket(const std::string& bundleName,
                                          const RuntimeAction& cpuAction,
                                          const RuntimeAction& gpuAction,
                                          const RuntimeAction& postGpuAction) {
    Logger::instance().log("[Runtime] Start CPU/GPU/Post-GPU action bundle");

    if        (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                               "Given CPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[CudaRuntime::executeTasks_FullPacket] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                               "Given GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[CudaRuntime::executeTasks_FullPacket] "
                                    "Need at least one tile per GPU packet");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                               "Given post-GPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[CudaRuntime::executeTasks_FullPacket] "
                                    "Post-GPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
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
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
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
    std::shared_ptr<DataPacket>       packet_gpu = std::shared_ptr<DataPacket>{};
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        // If we create a first shared_ptr and enqueue it with one team, it is
        // possible that this shared_ptr could have the action applied to its
        // data and go out of scope before we create a second shared_ptr.  In
        // this case, the data item's resources would be released prematurely.
        // To avoid this, we create all copies up front and before enqueing any
        // copy.
        tile_cpu = ti->buildCurrentTile();
        tile_gpu = tile_cpu;
        if (   (tile_cpu.get() != tile_gpu.get())
            || (tile_cpu.use_count() != 2)) {
            throw std::runtime_error("tile_cpu and tile_gpu not matched");
        }

        packet_gpu = std::shared_ptr<DataPacket>{ new CudaDataPacket{std::move(tile_gpu)} };
        if (   (tile_gpu != nullptr)
            || (tile_gpu.use_count() != 0)) {
            throw std::runtime_error("tile_gpu not nulled");
//        } else if (   (packet_gpu->getTile().get() != tile_cpu.get())
//                   || (tile_cpu.use_count() != 2)) {
//            throw std::runtime_error("tile_cpu and packet_gpu not matched");
        }

        // CPU action parallel pipeline
        cpuTeam->enqueue( std::move(tile_cpu) );
        if (   (tile_cpu != nullptr)
            || (tile_cpu.use_count() != 0)) {
            throw std::runtime_error("tile_cpu not nulled");
        }

        // GPU/Post-GPU action parallel pipeline
        packet_gpu->initiateHostToDeviceTransfer();
        gpuTeam->enqueue( std::move(packet_gpu) );
        if (   (packet_gpu != nullptr)
            || (packet_gpu.use_count() != 0)) {
            throw std::runtime_error("packet_gpu not nulled");
        }
    }
    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    // host thread blocks until cycle ends
    cpuTeam->wait();
    gpuTeam->wait();
    postGpuTeam->wait();

    //***** BREAK APART THREAD TEAM CONFIGURATION
    cpuTeam->detachThreadReceiver();
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();
    gpuToHost_.detachDataReceiver();

    Logger::instance().log("[CudaRuntime] End CPU/GPU action");
}
#endif

}

