#include "testThreadRoutines.h"

#include <unistd.h>
#include <thread>
#include <chrono>

void TestThreadRoutines::noop(const int tId, orchestration::DataItem* dataItem) {  }

void TestThreadRoutines::delay_10ms(const int tId, orchestration::DataItem* dataItem) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void TestThreadRoutines::delay_100ms(const int tId, orchestration::DataItem* dataItem) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

