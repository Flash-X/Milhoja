#ifndef TEST_THREAD_ROUTINES_H__
#define TEST_THREAD_ROUTINES_H__

#include <string>

namespace TestThreadRoutines {
    void noop(const unsigned int tId, const std::string& name, unsigned int work);
    void delay_500ms(const unsigned int tId, const std::string& name, unsigned int work);
}

#endif

