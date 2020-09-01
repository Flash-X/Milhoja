#ifndef TIMERS_H__
#define TIMERS_H__

#include "Timer.h"

namespace orchestration {

class Timers {
    public:
    static void start(std::string name);
    static void stop(std::string name);

    static std::shared_ptr<Timer> current_;
    static const std::shared_ptr<Timer> base_;
};

}

#endif

