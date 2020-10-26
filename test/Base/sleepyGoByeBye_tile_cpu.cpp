#include "sleepyGoByeBye.h"

#include <chrono>
#include <thread>

void ActionRoutines::sleepyGoByeBye_tile_cpu(const int tId,
                                             orchestration::DataItem* dataItem) {
    std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_US));  
}

