#include "Timers.h"

namespace orchestration {

// initialize static members
const std::shared_ptr<Timer> Timers::base_ = std::make_shared<Timer>();
std::shared_ptr<Timer> Timers::current_ = base_;

void Timers::start(std::string name) {
    Timers::current_ = Timers::current_->getChild(name);
    Timers::current_->start();
}

void Timers::stop(std::string name) {
    if(Timers::current_->name() == name) {
        Timers::current_->stop();
    } else {
        throw std::logic_error("Timers::stop called out-of-order.");
    }

    Timers::current_ = Timers::current_->parent();
}

std::string Timers::getSummary() {
    std::stringstream ss;
    base_->makeSummary(ss, 0);
    return ss.str();
}

}

