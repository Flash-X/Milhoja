#ifndef DATA_ITEM_SPLITTER_H__
#define DATA_ITEM_SPLITTER_H__

#include "RuntimeElement.h"

namespace orchestration {

class DataItemSplitter : public RuntimeElement {
public:
    DataItemSplitter(void)    { };
    ~DataItemSplitter(void)   { };

    void increaseThreadCount(const unsigned int nThreads) override;

    void enqueue(std::shared_ptr<DataItem>&& dataItem) override;
    void closeQueue(void) override;

private:
    DataItemSplitter(DataItemSplitter&) = delete;
    DataItemSplitter(const DataItemSplitter&) = delete;
    DataItemSplitter(DataItemSplitter&&) = delete;
    DataItemSplitter& operator=(DataItemSplitter&) = delete;
    DataItemSplitter& operator=(const DataItemSplitter&) = delete;
    DataItemSplitter& operator=(DataItemSplitter&&) = delete;
};

}

#endif

