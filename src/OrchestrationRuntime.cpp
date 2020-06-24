#include "OrchestrationRuntime.h"

#include <stdexcept>
#include <iostream>

#include "Grid.h"
#include "Flash.h"

std::string     OrchestrationRuntime::logFilename_       = "";
unsigned int    OrchestrationRuntime::nTileTeams_        = 0;
unsigned int    OrchestrationRuntime::nPacketTeams_      = 0;
unsigned int    OrchestrationRuntime::maxThreadsPerTeam_ = 0;
bool            OrchestrationRuntime::instantiated_      = false;

/**
 * 
 *
 * \return 
 */
OrchestrationRuntime& OrchestrationRuntime::instance(void) {
    instantiated_ = true;
    static OrchestrationRuntime     orSingleton;
    return orSingleton;
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::setLogFilename(const std::string& filename) {
    if (instantiated_) {
        throw std::logic_error("[OrchestrationRuntime::setLogFilename] "
                               "Set only when runtime does not exist");
    }

    logFilename_ = filename;
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::setNumberThreadTeams(const unsigned int nTileTeams,
                                                const unsigned int nPacketTeams) {
    if (instantiated_) {
        throw std::logic_error("[OrchestrationRuntime::setNumberThreadTeams] "
                               "Set only when runtime does not exist");
    } else if ((nTileTeams == 0) && (nPacketTeams == 0)) {
        throw std::invalid_argument("[OrchestrationRuntime::setNumberThreadTeams] "
                                    "Need at least one ThreadTeam");
    }

    nTileTeams_   = nTileTeams;
    nPacketTeams_ = nPacketTeams;
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::setMaxThreadsPerTeam(const unsigned int nThreads) {
    if (instantiated_) {
        throw std::logic_error("[OrchestrationRuntime::setMaxThreadsPerTeam] "
                               "Set only when runtime does not exist");
    } else if (nThreads == 0) {
        throw std::invalid_argument("[OrchestrationRuntime::setMaxThreadsPerTeam] "
                                    "Need at least one thread per team");
    }

    maxThreadsPerTeam_ = nThreads;
}

/**
 * 
 *
 * \return 
 */
OrchestrationRuntime::OrchestrationRuntime(void) {
#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Initializing\n";
    logFile_.close();
#endif

    if ((nTileTeams_ <= 0) && (nPacketTeams_ <= 0)) {
        throw std::invalid_argument("[OrchestrationRuntime::OrchestrationRuntime] "
                                    "Need to create at least one team");
    }

    tileTeams_ = new ThreadTeam<Tile>*[nTileTeams_];
    for (unsigned int i=0; i<nTileTeams_; ++i) {
        tileTeams_[i] = new ThreadTeam<Tile>(maxThreadsPerTeam_, i, logFilename_);
    }

    packetTeams_ = new ThreadTeam<DataPacket>*[nPacketTeams_];
    for (unsigned int i=0; i<nPacketTeams_; ++i) {
        packetTeams_[i] = new ThreadTeam<DataPacket>(maxThreadsPerTeam_, i, logFilename_);
    }

#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Initialized\n";
    logFile_.close();
#endif
}

/**
 * 
 *
 * \return 
 */
OrchestrationRuntime::~OrchestrationRuntime(void) {
    instantiated_ = false;

#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Finalizing\n";
    logFile_.close();
#endif

    for (unsigned int i=0; i<nTileTeams_; ++i) {
        delete tileTeams_[i];
        tileTeams_[i] = nullptr;
    }
    delete [] tileTeams_;
    tileTeams_ = nullptr;

    for (unsigned int i=0; i<nPacketTeams_; ++i) {
        delete packetTeams_[i];
        packetTeams_[i] = nullptr;
    }
    delete [] packetTeams_;
    packetTeams_ = nullptr;

#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Finalized\n";
    logFile_.close();
#endif
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeTasks(const ActionBundle& bundle) {
#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Start execution of " 
             << bundle.name << std::endl;
    logFile_.close();
#endif

    if      (bundle.distribution != WorkDistribution::Concurrent) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks] "
                               "The runtime only handles concurrent work distriution");
    }

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
    } else if (!hasCpuAction && hasGpuAction && !hasPostGpuAction) {
        executeGpuTasks(bundle.name,
                        bundle.gpuAction);
    } else if (hasCpuAction && hasGpuAction && !hasPostGpuAction) {
        executeConcurrentCpuGpuTasks(bundle.name, 
                                     bundle.cpuAction,
                                     bundle.gpuAction);
    } else if (   hasCpuAction && hasGpuAction && hasPostGpuAction
               && bundle.cpuAction.teamType     == ThreadTeamDataType::BLOCK
               && bundle.gpuAction.teamType     == ThreadTeamDataType::BLOCK
               && bundle.postGpuAction.teamType == ThreadTeamDataType::BLOCK) {
        executeTasks_Full(bundle.name,
                          bundle.cpuAction,
                          bundle.gpuAction,
                          bundle.postGpuAction);
    } else if (   hasCpuAction && hasGpuAction && hasPostGpuAction
               && bundle.cpuAction.teamType     == ThreadTeamDataType::BLOCK 
               && bundle.gpuAction.teamType     == ThreadTeamDataType::SET_OF_BLOCKS
               && bundle.postGpuAction.teamType == ThreadTeamDataType::SET_OF_BLOCKS) {
        executeTasks_FullPacket(bundle.name,
                                bundle.cpuAction,
                                bundle.gpuAction,
                                bundle.postGpuAction);
    } else {
        std::string   errMsg =   "[OrchestrationRuntime::executeTasks] ";
        errMsg += "No compatible thread team layout - ";
        errMsg += bundle.name;
        errMsg += "\n";
        throw std::logic_error(errMsg);
    }

#ifdef DEBUG_RUNTIME
    logFile_.open(logFilename_, std::ios::out | std::ios::app);
    logFile_ << "[OrchestrationRuntime] Finished execution of " 
             << bundle.name << std::endl;
    logFile_.close();
#endif
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeTasks_Full(const std::string& bundleName,
                                             const RuntimeAction& cpuAction,
                                             const RuntimeAction& gpuAction,
                                             const RuntimeAction& postGpuAction) {
    if      (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeTasks_Full] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                               "Given GPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeTasks_Full] "
                                    "GPU tiles/packet should be zero since it is tile-based");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                               "Given post-GPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeTasks_Full] "
                                    "Post-GPU tiles/packet should be zero since it is tile-based");
    } else if (nTileTeams_ < 3) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                               "Need at least three tile ThreadTeams in runtime");
    }

    ThreadTeam<Tile>*   cpuTeam     = tileTeams_[0];
    ThreadTeam<Tile>*   gpuTeam     = tileTeams_[1];
    ThreadTeam<Tile>*   postGpuTeam = tileTeams_[2];

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_Full] "
                                "Post-GPU could receive too many threads "
                                "from the CPU and GPU teams");
    }

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachWorkReceiver(postGpuTeam);

    cpuTeam->startTask(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startTask(gpuAction, "Concurrent_GPU_Block_Team");
    postGpuTeam->startTask(postGpuAction, "Post_GPU_Block_Team");

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread pools.  When a work unit is finished on the GPU, the work unit
    // shall be enqueued automatically for the post-GPU pool.
    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        Tile  work(itor, level);
        // Ownership of tile resources is transferred to last team
        cpuTeam->enqueue(work, false);
        gpuTeam->enqueue(work, true);
    }
    gpuTeam->closeTask();
    cpuTeam->closeTask();

    // TODO: We could give subscribers a pointer to the publisher so that during
    // the subscriber's wait() it can determine if it should terminate yet.  The
    // goal of this would be to allow client code to call the wait() methods in
    // any order.  I doubt that the design/code complexity is worth this minor
    // gain.

    // CPU and GPU pools are not dependent on any other pools
    //   => call these first
    cpuTeam->wait();
    cpuTeam->detachThreadReceiver();

    gpuTeam->wait();
    
    // The GPU pool has no more threads or work to push to its dependents
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachWorkReceiver();

    // Post-GPU follows GPU in the thread pool work/thread pipeline
    //   => the gpuTeam wait() method *must* terminate before the postGpuTeam
    //      wait() method is called to ensure that all GPU work are queued
    //      in postGpuTeam before the post-GPU thread pool can begin
    //      determining if it should terminate.
    postGpuTeam->wait();
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeTasks_FullPacket(const std::string& bundleName,
                                                   const RuntimeAction& cpuAction,
                                                   const RuntimeAction& gpuAction,
                                                   const RuntimeAction& postGpuAction) {
    if        (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_FullPacket] "
                               "Given CPU action should run on tile-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeTasks_FullPacket] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_FullPacket] "
                               "Given GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeTasks_FullPacket] "
                                    "Need at least one tile per GPU packet");
    } else if (postGpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_FullPacket] "
                               "Given post-GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeTasks_FullPacket] "
                                    "Post-GPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTileTeams_ < 1) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_FullPacket] "
                               "Need at least one tile ThreadTeam in runtime");
    } else if (nPacketTeams_ < 2) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_FullPacket] "
                               "Need at least two packet ThreadTeams in runtime");
    }

    ThreadTeam<Tile>*         cpuTeam     = tileTeams_[0];
    ThreadTeam<DataPacket>*   gpuTeam     = packetTeams_[0];
    ThreadTeam<DataPacket>*   postGpuTeam = packetTeams_[1];

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[OrchestrationRuntime::executeTasks_FullPacket] "
                                "Post-GPU could receive too many thread "
                                "activation calls from CPU and GPU teams");
    }

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachWorkReceiver(postGpuTeam);

    cpuTeam->startTask(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startTask(gpuAction, "Concurrent_GPU_Packet_Team");
    postGpuTeam->startTask(postGpuAction, "Post_GPU_Packet_Team");

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread teams.  When a data item is finished on the GPU, the data item
    // is enqueued automatically with the post-GPU team.
    DataPacket     gpuPacket;
    unsigned int   level = 0;
    Grid&          grid = Grid::instance();
    gpuPacket.clear();
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        // TODO: What is the best way to manage the copy/move actions here?
        //       I am just playing around at the moment.
        Tile  work(itor, level);
        cpuTeam->enqueue(work, false);
        gpuPacket.tileList.push_front(std::move(work));

        if (gpuPacket.tileList.size() >= gpuAction.nTilesPerPacket) {
            gpuTeam->enqueue(gpuPacket, true);
            gpuPacket.clear();
        }
    }

    if (gpuPacket.tileList.size() != 0) {
        gpuTeam->enqueue(gpuPacket, true);
    }

    gpuTeam->closeTask();
    cpuTeam->closeTask();

    gpuPacket.clear();

    // TODO: We could give subscribers a pointer to the publisher so that during
    // the subscriber's wait() it can determine if it should terminate yet.  The
    // goal of this would be to allow client code to call the wait() methods in
    // any order.  I doubt that the design/code complexity is worth this minor
    // gain.

    cpuTeam->wait();
    cpuTeam->detachThreadReceiver();

    gpuTeam->wait();
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachWorkReceiver();

    postGpuTeam->wait();
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeConcurrentCpuGpuTasks(const std::string& bundleName,
                                                        const RuntimeAction& cpuAction,
                                                        const RuntimeAction& gpuAction) {
    if      (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeConcurrentCpuGpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeConcurrentCpuGpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeConcurrentCpuGpuTasks] "
                               "Given GPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeConcurrentCpuGpuTasks] "
                                    "GPU tiles/packet should be zero since it is tile-based");
    } else if (nTileTeams_ < 2) {
        throw std::logic_error("[OrchestrationRuntime::executeConcurrentCpuGpuTasks] "
                               "Need at least two tile ThreadTeams in runtime");
    }

    ThreadTeam<Tile>*   cpuTeam = tileTeams_[0];
    ThreadTeam<Tile>*   gpuTeam = tileTeams_[1];

    cpuTeam->startTask(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startTask(gpuAction, "Concurrent_GPU_Block_Team");

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        Tile  work(itor, level);
        // Ownership of tile resources is transferred to last team
        cpuTeam->enqueue(work, false);
        gpuTeam->enqueue(work, true);
    }
    gpuTeam->closeTask();
    cpuTeam->closeTask();

    cpuTeam->wait();
    gpuTeam->wait();
}

/**
 * 
 *
 * \return 
 */
void OrchestrationRuntime::executeCpuTasks(const std::string& bundleName,
                                           const RuntimeAction& cpuAction) {
    if (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[OrchestrationRuntime::executeCpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeCpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (nTileTeams_ < 1) {
        throw std::logic_error("[OrchestrationRuntime::executeCpuTasks] "
                               "Need at least one ThreadTeam in runtime");
    }

    ThreadTeam<Tile>*   cpuTeam = tileTeams_[0];

    cpuTeam->startTask(cpuAction, "CPU_Block_Team");

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        Tile  work(itor, level);
        cpuTeam->enqueue(work, true);
    }
    cpuTeam->closeTask();
    cpuTeam->wait();
}

void OrchestrationRuntime::executeGpuTasks(const std::string& bundleName,
                                           const RuntimeAction& gpuAction) {
    if (gpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[OrchestrationRuntime::executeGpuTasks] "
                               "Given GPU action should run on a thread team "
                               "that works with data packets of blocks");
    } else if (gpuAction.nTilesPerPacket <= 0) {
        throw std::invalid_argument("[OrchestrationRuntime::executeGpuTasks] "
                                    "Need at least one tile per packet");
    } else if (nPacketTeams_ < 1) {
        throw std::logic_error("[OrchestrationRuntime::executeGpuTasks] "
                               "Need at least one ThreadTeams in runtime");
    }

    ThreadTeam<DataPacket>*   gpuTeam = packetTeams_[0];

    DataPacket  packet;
    packet.clear();

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();

    gpuTeam->startTask(gpuAction, "GPU_PacketOfBlocks_Team");

    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        Tile  work(itor, level);
        packet.tileList.push_front(std::move(work));

        if (packet.tileList.size() >= gpuAction.nTilesPerPacket) {
            gpuTeam->enqueue(packet, true);
            packet.clear();
        }
    }

    if (packet.tileList.size() != 0) {
        gpuTeam->enqueue(packet, true);
        gpuTeam->closeTask();
        packet.clear();
    } else {
        gpuTeam->closeTask();
    }
    gpuTeam->wait();
}

