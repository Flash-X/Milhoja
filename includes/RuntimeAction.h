#ifndef RUNTIME_ACTION_H__
#define RUNTIME_ACTION_H__

#include "ThreadTeamDataType.h"

// TODO: We need to create the concept of an action chain
//       These are a sequence of RuntimeActions with consecutive teams 
//       linked as publisher/subscriber.  The first team in the chain
//       is fed data items by a work distributor.
struct RuntimeAction {
    std::string           name = "NoName";
    unsigned int          nInitialThreads = 0;
    ThreadTeamDataType    teamType = ThreadTeamDataType::BLOCK;
    TASK_FCN              routine = nullptr;
};

#endif

