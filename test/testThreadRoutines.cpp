#include "testThreadRoutines.h"

#include <unistd.h>
#include <thread>
#include <chrono>

void TestThreadRoutines::noop(const unsigned int tId,
                              const std::string& name,
                              unsigned int work) {
    return;
}

void TestThreadRoutines::delay_500ms(const unsigned int tId,
                                     const std::string& name,
                                     unsigned int work) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return;
}

