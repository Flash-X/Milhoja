#ifndef TEST_THREAD_ROUTINES_H__
#define TEST_THREAD_ROUTINES_H__

#include <Milhoja_DataItem.h>

namespace TestThreadRoutines {
    void noop(const int tId, milhoja::DataItem* dataItem);
    void delay_10ms(const int tId, milhoja::DataItem* dataItem);
    void delay_100ms(const int tId, milhoja::DataItem* dataItem);
}

#endif

