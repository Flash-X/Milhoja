#ifndef DATA_ITEM_H__
#define DATA_ITEM_H__

namespace orchestration {

class DataItem {
public:
    virtual ~DataItem(void) {}

    DataItem(DataItem&)                  = delete;
    DataItem(const DataItem&)            = delete;
    DataItem(DataItem&&)                 = delete;
    DataItem& operator=(DataItem&)       = delete;
    DataItem& operator=(const DataItem&) = delete;
    DataItem& operator=(DataItem&&)      = delete;
protected:
    DataItem(void) {}
};

}

#endif

