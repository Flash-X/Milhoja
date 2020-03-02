#ifndef TEST_THREAD_ROUTINES_H__
#define TEST_THREAD_ROUTINES_H__

#include <string>

namespace TestThreadRoutines {
    void noop(const unsigned int tId,
              const std::string& name,
              const int& work);
    void delay_10ms(const unsigned int tId,
                    const std::string& name,
                    const int& work);
    void delay_100ms(const unsigned int tId,
                     const std::string& name,
                     const int& work);
}

#endif

