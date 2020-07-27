#include "OrchestrationRuntime.h"

#include <stdexcept>
#include <iostream>

#include "Grid.h"
#include "OrchestrationLogger.h"
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
    instantiated_ = true;
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
Runtime::Runtime(void) {
#ifdef DEBUG_RUNTIME
    Logger::instance().log("[Runtime] Initializing");
#endif

    if (nTeams_ <= 0) {
        throw std::invalid_argument("[Runtime::Runtime] "
                                    "Need to create at least one team");
    }

    teams_ = new ThreadTeam*[nTeams_];
    for (unsigned int i=0; i<nTeams_; ++i) {
        teams_[i] = new ThreadTeam(maxThreadsPerTeam_, i);
    }

#ifdef DEBUG_RUNTIME
    Logger::instance().log("[Runtime] Initialized");
#endif
}

/**
 * 
 *
 * \return 
 */
Runtime::~Runtime(void) {
    instantiated_ = false;

#ifdef DEBUG_RUNTIME
    Logger::instance().log("[Runtime] Finalizing");
#endif

    for (unsigned int i=0; i<nTeams_; ++i) {
        delete teams_[i];
        teams_[i] = nullptr;
    }
    delete [] teams_;
    teams_ = nullptr;

#ifdef DEBUG_RUNTIME
    Logger::instance().log("[Runtime] Finalized");
#endif
}

/**
 * 
 *
 * \return 
 */
void Runtime::executeTasks(const ActionBundle& bundle) {
#ifdef DEBUG_RUNTIME
    std::string msg = "[Runtime] Start execution of " + bundle.name;
    Logger::instance().log(msg);
#endif

    if      (bundle.distribution != WorkDistribution::Concurrent) {
        throw std::logic_error("[Runtime::executeTasks] "
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
        std::string   errMsg =   "[Runtime::executeTasks] ";
        errMsg += "No compatible thread team layout - ";
        errMsg += bundle.name;
        errMsg += "\n";
        throw std::logic_error(errMsg);
    }

#ifdef DEBUG_RUNTIME
    msg  = "[Runtime] Finished execution of " + bundle.name;
    Logger::instance().log(msg);
#endif
}

/**
 * 
 *
 * \return 
 */
void Runtime::executeTasks_Full(const std::string& bundleName,
                                const RuntimeAction& cpuAction,
                                const RuntimeAction& gpuAction,
                                const RuntimeAction& postGpuAction) {
    if      (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeTasks_Full] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeTasks_Full] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeTasks_Full] "
                               "Given GPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeTasks_Full] "
                                    "GPU tiles/packet should be zero since it is tile-based");
    } else if (postGpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeTasks_Full] "
                               "Given post-GPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeTasks_Full] "
                                    "Post-GPU tiles/packet should be zero since it is tile-based");
    } else if (nTeams_ < 3) {
        throw std::logic_error("[Runtime::executeTasks_Full] "
                               "Need at least three ThreadTeams in runtime");
    }

    ThreadTeam*   cpuTeam     = teams_[0];
    ThreadTeam*   gpuTeam     = teams_[1];
    ThreadTeam*   postGpuTeam = teams_[2];

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeTasks_Full] "
                                "Post-GPU could receive too many threads "
                                "from the CPU and GPU teams");
    }

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachDataReceiver(postGpuTeam);

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread pools.  When a work unit is finished on the GPU, the work unit
    // shall be enqueued automatically for the post-GPU pool.
    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    std::shared_ptr<DataItem>  dataItem_cpu{};
    std::shared_ptr<DataItem>  dataItem_gpu{};
    if ((dataItem_cpu.get() != nullptr) || (dataItem_cpu.use_count() != 0)) {
        throw std::logic_error("CPU shared_ptr not NULLED at creation");
    }
    if ((dataItem_gpu.get() != nullptr) || (dataItem_gpu.use_count() != 0)) {
        throw std::logic_error("GPU shared_ptr not NULLED at creation");
    }

    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Block_Team");
    postGpuTeam->startCycle(postGpuAction, "Post_GPU_Block_Team");
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        // If we create a first shared_ptr and enqueue it with one team, it is
        // possible that this shared_ptr could have the action applied to its
        // data and go out of scope before we create a second shared_ptr.  In
        // this case, the data item's resources would be released prematurely.
        // To avoid this, we create all copies up front and before enqueing any
        // copy.
        dataItem_cpu = std::shared_ptr<DataItem>{ new Tile{itor, level} };
        dataItem_gpu = dataItem_cpu;
        if ((dataItem_cpu.get()) != (dataItem_gpu.get())) {
            throw std::logic_error("shared_ptr copy didn't work");
        } else if (dataItem_cpu.use_count() != 2) {
            throw std::logic_error("Unexpected shared_ptr count");
        }
//        std::cout << "[Runtime] Created " 
//                  << dataItem_cpu.use_count()
//                  << " copies of block "
//                  << dataItem_cpu->gridIndex()
//                  << " shared_ptr\n";

        // Move so that the shared ownership gets transferred with the data item
        // to the Thread Teams and the data items here gets nulled.  Therefore,
        // the shared_ptr counter should not change with these calls and
        // the release of the Tile resources does not depend on the shared_ptrs
        // in this scope.
        cpuTeam->enqueue(std::move(dataItem_cpu));
        gpuTeam->enqueue(std::move(dataItem_gpu));

        if ((dataItem_cpu.get() != nullptr) || (dataItem_cpu.use_count() != 0)) {
            throw std::logic_error("CPU shared_ptr not NULLED");
        }
        if ((dataItem_gpu.get() != nullptr) || (dataItem_gpu.use_count() != 0)) {
            throw std::logic_error("GPU shared_ptr not NULLED");
        }
    }
    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

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
    gpuTeam->detachDataReceiver();

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
void Runtime::executeTasks_FullPacket(const std::string& bundleName,
                                      const RuntimeAction& cpuAction,
                                      const RuntimeAction& gpuAction,
                                      const RuntimeAction& postGpuAction) {
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
    } else if (postGpuAction.teamType != ThreadTeamDataType::SET_OF_BLOCKS) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Given post-GPU action should run on packet-based "
                               "thread team, which is not in configuration");
    } else if (postGpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeTasks_FullPacket] "
                                    "Post-GPU should have zero tiles/packet as "
                                    "client code cannot control this");
    } else if (nTeams_ < 1) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                               "Need at least one ThreadTeam in runtime");
    }

    ThreadTeam*   cpuTeam     = teams_[0];
    ThreadTeam*   gpuTeam     = teams_[1];
    ThreadTeam*   postGpuTeam = teams_[2];

    unsigned int nTotalThreads =       cpuAction.nInitialThreads
                                 +     gpuAction.nInitialThreads
                                 + postGpuAction.nInitialThreads;
    if (nTotalThreads > postGpuTeam->nMaximumThreads()) {
        throw std::logic_error("[Runtime::executeTasks_FullPacket] "
                                "Post-GPU could receive too many thread "
                                "activation calls from CPU and GPU teams");
    }

    //***** Construct thread and work pipelines
    cpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachThreadReceiver(postGpuTeam);
    gpuTeam->attachDataReceiver(postGpuTeam);

    // Data is enqueued for both the concurrent CPU and concurrent GPU
    // thread teams.  When a data item is finished on the GPU, the data item
    // is enqueued automatically with the post-GPU team.
    unsigned int   level = 0;
    Grid&          grid = Grid::instance();

    std::shared_ptr<DataItem>   dataItem_cpu{};
    std::shared_ptr<DataItem>   dataItem_tmp{};
    std::shared_ptr<DataItem>   dataItem_gpu{ new DataPacket{} };

    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Packet_Team");
    postGpuTeam->startCycle(postGpuAction, "Post_GPU_Packet_Team");
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        dataItem_cpu = std::shared_ptr<DataItem>{ new Tile{itor, level} };
        dataItem_tmp = dataItem_cpu;
        dataItem_gpu->addSubItem( std::move(dataItem_tmp) );

        cpuTeam->enqueue( std::move(dataItem_cpu) );
        if (dataItem_gpu->nSubItems() >= gpuAction.nTilesPerPacket) {
            gpuTeam->enqueue( std::move(dataItem_gpu) );
            dataItem_gpu = std::shared_ptr<DataItem>{ new DataPacket{} };
        }
    }

    if (dataItem_gpu->nSubItems() > 0) {
        gpuTeam->enqueue( std::move(dataItem_gpu) );
    } else {
        dataItem_gpu.reset();
    }

    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    // TODO: We could give subscribers a pointer to the publisher so that during
    // the subscriber's wait() it can determine if it should terminate yet.  The
    // goal of this would be to allow client code to call the wait() methods in
    // any order.  I doubt that the design/code complexity is worth this minor
    // gain.

    cpuTeam->wait();
    cpuTeam->detachThreadReceiver();

    gpuTeam->wait();
    gpuTeam->detachThreadReceiver();
    gpuTeam->detachDataReceiver();

    postGpuTeam->wait();
}

/**
 * 
 *
 * \return 
 */
void Runtime::executeConcurrentCpuGpuTasks(const std::string& bundleName,
                                           const RuntimeAction& cpuAction,
                                           const RuntimeAction& gpuAction) {
    if      (cpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeConcurrentCpuGpuTasks] "
                               "Given CPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (cpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeConcurrentCpuGpuTasks] "
                                    "CPU tiles/packet should be zero since it is tile-based");
    } else if (gpuAction.teamType != ThreadTeamDataType::BLOCK) {
        throw std::logic_error("[Runtime::executeConcurrentCpuGpuTasks] "
                               "Given GPU action should run on block-based "
                               "thread team, which is not in configuration");
    } else if (gpuAction.nTilesPerPacket != 0) {
        throw std::invalid_argument("[Runtime::executeConcurrentCpuGpuTasks] "
                                    "GPU tiles/packet should be zero since it is tile-based");
    } else if (nTeams_ < 2) {
        throw std::logic_error("[Runtime::executeConcurrentCpuGpuTasks] "
                               "Need at least two ThreadTeams in runtime");
    }

    ThreadTeam*   cpuTeam = teams_[0];
    ThreadTeam*   gpuTeam = teams_[1];

    cpuTeam->startCycle(cpuAction, "Concurrent_CPU_Block_Team");
    gpuTeam->startCycle(gpuAction, "Concurrent_GPU_Block_Team");

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();

    std::shared_ptr<DataItem>   dataItem_cpu{};
    std::shared_ptr<DataItem>   dataItem_gpu{};
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        dataItem_cpu = std::shared_ptr<DataItem>{ new Tile{itor, level} };
        dataItem_gpu = dataItem_cpu;

        cpuTeam->enqueue( std::move(dataItem_cpu) );
        gpuTeam->enqueue( std::move(dataItem_gpu) );
    }
    gpuTeam->closeQueue();
    cpuTeam->closeQueue();

    cpuTeam->wait();
    gpuTeam->wait();
}

/**
 * 
 *
 * \return 
 */
void Runtime::executeCpuTasks(const std::string& bundleName,
                              const RuntimeAction& cpuAction) {
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

    ThreadTeam*   cpuTeam = teams_[0];

    cpuTeam->startCycle(cpuAction, "CPU_Block_Team");

    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        cpuTeam->enqueue( std::shared_ptr<DataItem>{ new Tile{itor, level} } );
    }
    cpuTeam->closeQueue();
    cpuTeam->wait();
}

void Runtime::executeGpuTasks(const std::string& bundleName,
                              const RuntimeAction& gpuAction) {
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

    ThreadTeam*   gpuTeam = teams_[0];

    // TODO: For this configuration, could I make the dataItems unique_ptrs?
    //       Would this work at the level of the ThreadTeam?  What about the
    //       case when there is a data parallel helper in the pipeline?
    //       It might lead to
    //       better performance and would be more explicit.  
    unsigned int   level = 0;
    Grid&   grid = Grid::instance();
    auto dataItem_gpu = std::shared_ptr<DataItem>{ new DataPacket{} };

    gpuTeam->startCycle(gpuAction, "GPU_PacketOfBlocks_Team");
    for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
        dataItem_gpu->addSubItem( std::shared_ptr<DataItem>{ new Tile{itor, level} } );

        if (dataItem_gpu->nSubItems() >= gpuAction.nTilesPerPacket) {
            gpuTeam->enqueue( std::move(dataItem_gpu) );
            dataItem_gpu = std::shared_ptr<DataItem>{ new DataPacket{} };
        }
    }

    if (dataItem_gpu->nSubItems() != 0) {
        gpuTeam->enqueue( std::move(dataItem_gpu) );
        gpuTeam->closeQueue();
    } else {
        gpuTeam->closeQueue();
        dataItem_gpu.reset();
    }
    gpuTeam->wait();
}

}

