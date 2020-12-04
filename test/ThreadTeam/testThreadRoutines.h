#ifndef TEST_THREAD_ROUTINES_H__
#define TEST_THREAD_ROUTINES_H__

#include <string>

#include "DataItem.h"

namespace TestThreadRoutines {
    void noop(const int tId, orchestration::DataItem* dataItem);
    void delay_10ms(const int tId, orchestration::DataItem* dataItem);
    void delay_100ms(const int tId, orchestration::DataItem* dataItem);
}

#endif

