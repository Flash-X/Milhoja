#include "testThreadRoutines.h"

#include <unistd.h>
#include <thread>
#include <chrono>

void TestThreadRoutines::noop(const unsigned int tId,
                              const std::string& name,
                              const int& work) {  }

void TestThreadRoutines::delay_10ms(const unsigned int tId,
                                    const std::string& name,
                                    const int& work) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void TestThreadRoutines::delay_100ms(const unsigned int tId,
                                     const std::string& name,
                                     const int& work) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

