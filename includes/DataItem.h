#ifndef DATA_ITEM_H__
#define DATA_ITEM_H__

#include <memory>

#include "CudaStream.h"

namespace orchestration {

class DataItem {
public:
    virtual ~DataItem(void) {}

    virtual std::shared_ptr<DataItem>  getTile(void) = 0;
    virtual CudaStream&                stream(void) = 0;

    virtual void        initiateHostToDeviceTransfer(void) = 0;
    virtual void        transferFromDeviceToHost(void) = 0;

protected:
    DataItem(void) {}

private:
    DataItem(DataItem&) = delete;
    DataItem(const DataItem&) = delete;
    DataItem(DataItem&&) = delete;
    DataItem& operator=(DataItem&) = delete;
    DataItem& operator=(const DataItem&) = delete;
    DataItem& operator=(DataItem&&) = delete;
};

}

#endif

