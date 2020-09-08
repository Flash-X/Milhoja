#include "Timers.h"

#ifndef TIMERS_THREADED_TREE
#include <stack>
#endif

namespace orchestration {

// initialize static members
const std::shared_ptr<Timer> Timers::base_ = std::make_shared<Timer>();
std::shared_ptr<Timer> Timers::current_ = base_;
int Timers::maxDepth_ = 5; // number of levels possible in timer tree

void Timers::start(std::string name) {
    if(current_->depth() > Timers::maxDepth_) {
        throw std::logic_error("Trying to create Timer past max depth.");
    }
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
#ifdef TIMERS_THREADED_TREE
    std::shared_ptr<Timer> currentTimer = base_;
    while( currentTimer->getNext() )  {
        currentTimer = currentTimer->getNext();
        ss << currentTimer->getSummary();
    }
#else
    std::stack<std::shared_ptr<Timer>> timerStack;
    timerStack.push(base_);
    while( !timerStack.empty() ) {
        auto currentTimer = timerStack.top();
        timerStack.pop();

        auto childList = currentTimer->children();
        for(int i=childList.size()-1; i>=0; --i) {
            timerStack.push(childList[i]);
        }

        if(currentTimer!=base_) {
            ss << currentTimer->getSummary();
        }

    }
#endif
    return ss.str();
}

void Timers::setMaxDepth(const int num) {
    maxDepth_ = num;
}

}

