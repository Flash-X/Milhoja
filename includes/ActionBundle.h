#ifndef ACTION_BUNDLE_H__
#define ACTION_BUNDLE_H__

#include "RuntimeAction.h"

namespace orchestration {

enum class WorkDistribution     {Concurrent, Partitioned};

// WIP: This is just a rough draft to get things working.  This could
//      eventually be far more generic and flexible.
struct ActionBundle {
    std::string       name = "NoNameBundle";
    WorkDistribution  distribution = WorkDistribution::Concurrent;
    RuntimeAction     cpuAction;
    RuntimeAction     gpuAction;
    RuntimeAction     postGpuAction;
};

}

#endif

