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

    // Move over all subitems as individual elements and then
    // remove the now empty data packet from circulation and so that
    // the calling code has a nulled dataItem as expected.
    while (dataItem->nSubItems() > 0) {
        dataReceiver_->enqueue( dataItem->popSubItem() );
    }
    dataItem.reset();
}

void DataItemSplitter::closeQueue(void) {
    if (!dataReceiver_) {
        throw std::runtime_error("[DataItemSplitter::closeQueue] "
                                 "No data subscriber attached");
    }

    dataReceiver_->closeQueue();
}

}

