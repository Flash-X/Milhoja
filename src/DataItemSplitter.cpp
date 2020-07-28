#include "DataItemSplitter.h"

#include <iostream>

namespace orchestration {

void DataItemSplitter::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[DataItemSplitter::increaseThreadCount] "
                           "DataItemSplitters do no have threads to awaken");
}

void DataItemSplitter::enqueue(std::shared_ptr<DataItem>&& dataItem) {
    if (!dataReceiver_) {
        throw std::runtime_error("[DataItemSplitter::enqueue] "
                                 "No data subscriber attached");
    }

    // TODO: Just forward along for now.  Program in the splitting later.
    std::cout << "[DataItemSplitter::enqueue] I'm enqueuing with my subscriber\n";
    dataReceiver_->enqueue( std::move(dataItem) );
}

void DataItemSplitter::closeQueue(void) {
    if (!dataReceiver_) {
        throw std::runtime_error("[DataItemSplitter::closeQueue] "
                                 "No data subscriber attached");
    }

    std::cout << "[DataItemSplitter::closeQueue] I'm closing my subscriber's queue\n";
    dataReceiver_->closeQueue();
}

}

