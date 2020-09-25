#include "CudaRuntime.h"

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>

#include "ThreadTeam.h"
#include "Grid.h"
#include "OrchestrationLogger.h"
#include "MoverUnpacker.h"

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

unsigned int CudaRuntime::numberFreeStreams(void) const {
    return CudaStreamManager::instance().numberFreeStreams();
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::executeCpuTasks(const std::string& actionName,
                                  const RuntimeAction& cpuAction) {
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

    ThreadTeam*   cpuTeam = teams_[0];

    Logger::instance().log("[CudaRuntime] Start single CPU action");

    cpuTeam->startCycle(cpuAction, "CPU_Block_Team");

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        cpuTeam->enqueue( ti->buildCurrentTile() );
    }
    cpuTeam->closeQueue();
    cpuTeam->wait();

    Logger::instance().log("[CudaRuntime] End single CPU action");
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::executeGpuTasks(const std::string& bundleName,
                                  const RuntimeAction& gpuAction) {
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

    ThreadTeam*   gpuTeam = teams_[0];

    Logger::instance().log("[CudaRuntime] Start single GPU action");

    MoverUnpacker     gpuToHost{};
    gpuTeam->attachDataReceiver(&gpuToHost);

    // TODO: Good idea to call this as early as possible so that all threads
    //       go to wait while this this is setting everything up?  Hides
    //       TT wind-up cost.
    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");

    // We allocate the CUD memory here and pass the pointers around
    // with the data packet so that a work subscriber will
    // eventually deallocate the memory.
    auto packet_gpu = std::shared_ptr<CudaDataPacket>{};
    assert(packet_gpu == nullptr);
    assert(packet_gpu.use_count() == 0);

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        packet_gpu = std::make_shared<CudaDataPacket>( ti->buildCurrentTile() );
        packet_gpu->initiateHostToDeviceTransfer();

        gpuTeam->enqueue( std::move(packet_gpu) );
        assert(packet_gpu == nullptr);
        assert(packet_gpu.use_count() == 0);
    }

    gpuTeam->closeQueue();
    gpuTeam->wait();

    gpuTeam->detachDataReceiver();

    Logger::instance().log("[CudaRuntime] End single GPU action");
}

/**
 * 
 *
 * \return 
 */
void CudaRuntime::executeTasks_FullPacket(const std::string& bundleName,
                                          const RuntimeAction& cpuAction,
                                          const RuntimeAction& gpuAction,
                                          const RuntimeAction& postGpuAction) {
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

    ThreadTeam*        cpuTeam     = teams_[0];
    ThreadTeam*        gpuTeam     = teams_[1];
    ThreadTeam*        postGpuTeam = teams_[2];
    MoverUnpacker      gpuToHost{};

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[CudaRuntime::executeTasks_FullPacket] "
                                "Post-GPU could receive too many thread "
                                "activation calls from CPU and GPU teams");
    }

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachDataReceiver(&gpuToHost);
    gpuToHost.attachDataReceiver(postGpuTeam);

    std::shared_ptr<Tile>   tile_cpu{};
    std::shared_ptr<Tile>   tile_gpu{};
    auto                    packet_gpu = std::shared_ptr<CudaDataPacket>{};

    Logger::instance().log("[CudaRuntime] Start CPU/GPU action");

    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    postGpuTeam->startCycle(postGpuAction, "Post_GPU_Packet_Team");

    unsigned int   level = 0;
    Grid&          grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        // Acquire all resources before transferring ownership via enqueue
        // of any single resource.
        tile_cpu = ti->buildCurrentTile();
        tile_gpu = tile_cpu;
        assert(tile_cpu.get() == tile_gpu.get());
        assert(tile_cpu.use_count() == 2);
        packet_gpu = std::make_shared<CudaDataPacket>( std::move(tile_gpu) );
        assert(tile_gpu == nullptr);
        assert(tile_gpu.use_count() == 0);
        assert(packet_gpu.getTile().get() == tile_cpu.get());
        assert(tile_cpu.use_count() == 2);

        cpuTeam->enqueue( std::move(tile_cpu) );
        assert(tile_cpu == nullptr);
        assert(tile_cpu.use_count() == 0);

        packet_gpu->initiateHostToDeviceTransfer();
        gpuTeam->enqueue( std::move(packet_gpu) );
        assert(packet_gpu == nullptr);
        assert(packet_gpu.use_count() == 0);
    }
    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    // TODO: We could give subscribers a pointer to the publisher so that during
    // the subscriber's wait() it can determine if it should terminate yet.  The
    // goal of this would be to allow client code to call the wait() methods in
    // any order.  I doubt that the design/code complexity is worth this minor
    // gain.

    cpuTeam->wait();
    gpuTeam->wait();
    postGpuTeam->wait();

    Logger::instance().log("[CudaRuntime] End CPU/GPU action");

    cpuTeam->detachThreadReceiver();
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();
    gpuToHost.detachDataReceiver();
}

}

