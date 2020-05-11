#ifndef RUNTIME_ACTION_H__
#define RUNTIME_ACTION_H__

#include "ThreadTeamDataType.h"
#include "runtimeTask.h"

// TODO: We need to create the concept of an action chain
//       These are a sequence of RuntimeActions with consecutive teams 
//       linked as publisher/subscriber.  The first team in the chain
//       is fed data items by a work distributor.
struct RuntimeAction {
    std::string           name = "NoName";
    unsigned int          nInitialThreads = 0;
    ThreadTeamDataType    teamType = ThreadTeamDataType::BLOCK;
    // Client code should be able to specify a different number of tiles/packet
    // for each action.  A motivating example is an action bundle that will
    // execute one action using a packet-based team that will drive a GPU
    // and another action using another packet-based team that will drive an
    // FPGA.  It's possible that using one packet size for the FPGA and distinct
    // size for the GPU packet will be necessary to get the best performance.
    unsigned int          nTilesPerPacket = 0;
    TASK_FCN              routine = nullptr;
};

#endif

