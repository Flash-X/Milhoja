#ifndef TIMER_H__
#define TIMER_H__

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <stdexcept>

#include <iostream>

#include "Grid_REAL.h"

namespace orchestration {

// https://stackoverflow.com/questions/11711034/stdshared-ptr-of-this
class Timer : public std::enable_shared_from_this<Timer> {
public:
    // for making base node of Timers tree
    Timer()
        : name_{""},
          count_{0},
          depth_{0},
          running_{false},
          elapsedTime_{std::chrono::seconds::zero()},
          startTime_{}
    {
        //std::cout << "Constructor for base Timer." << std::endl;
    }

    // for making children
    Timer(std::string name, std::shared_ptr<Timer> parent)
        : parent_{parent},
          name_{name},
          count_{0},
          depth_{parent->depth()},
          running_{false},
          elapsedTime_{std::chrono::seconds::zero()},
          startTime_{}
    {
        //std::cout << "Constructor for Timer: " << name_ << std::endl;
    }

    ~Timer(void){
        //std::cout << "Destructor for Timer: " << name_ << std::endl;
    }

    Timer(Timer&&) = delete;
    Timer& operator=(Timer&&) = delete;
    Timer(Timer&) = delete;
    Timer(const Timer&) = delete;
    Timer& operator=(Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    std::shared_ptr<Timer> parent() const { return parent_.lock(); }
    std::string name() const { return name_; }
    int count() const { return count_; }
    int depth() const { return depth_; }
    bool running() const { return running_; }

    void start() {
        if(running_) {
            throw std::logic_error("Timer: " + name_ + " already running.");
        }

        ++count_;
        startTime_ =  std::chrono::steady_clock::now();
        running_ = true;

    }

    void stop() {
        if(!running_) {
            throw std::logic_error("Timer: " + name_ + " is not running.");
        }


        for(int i=0; i<children_.size(); ++i) {
            if( children_[i]->running() ) {
                children_[i]->stop();
                // or throw error?
            }
        }

        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
        elapsedTime_ += std::chrono::duration_cast<std::chrono::seconds>
                            (end - startTime_);

        running_ = false;
    }

    // gets desired child (or creates it if necessary)
    std::shared_ptr<Timer> getChild(std::string name) {
        int num = childNum(name);
        if (num>=0 && num<children_.size() ) {
            return children_[num];
        }
        else if(num==-1) {
            children_.push_back(
                    std::make_shared<Timer>(name,shared_from_this()) );
            int nchild = children_.size();
            return children_[nchild-1];
        }
        else {
            throw std::logic_error("Something went wrong in getChild.");
        }
    }

    void makeSummary(std::stringstream& ss, const int indent) const {
        int new_indent = indent;
        if(parent_.lock()) {
            for(int n=0; n<indent; ++n) {
                ss << " ";
            }
            ss << name_;
            ss << std::endl;
            new_indent += 2;
        }
        for(int i=0; i<children_.size(); ++i) {
            children_[i]->makeSummary(ss, new_indent);
        }
    }


private:
    int childNum(std::string name) const {
        int num = -1;
        for(int i=0; i<children_.size(); ++i) {
            if( children_[i]->name() == name) {
                num = i;
                break;
            }
        }
        return num;
    }

    std::weak_ptr<Timer> parent_;
    std::vector<std::shared_ptr<Timer>> children_;

    std::string name_;
    int count_;
    int depth_;
    bool running_;
    std::chrono::seconds elapsedTime_;
    std::chrono::steady_clock::time_point startTime_;
};

}

#endif

