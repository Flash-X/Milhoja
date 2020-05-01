#ifndef TEST_THREAD_ROUTINES_H__
#define TEST_THREAD_ROUTINES_H__

#include <string>

namespace TestThreadRoutines {
    void noop(const int tId, void* dataItem);
    void delay_10ms(const int tId, void* dataItem);
    void delay_100ms(const int tId, void* dataItem);
}

#endif

