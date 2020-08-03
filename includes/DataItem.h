#ifndef DATA_ITEM_H__
#define DATA_ITEM_H__

namespace orchestration {

class DataItem {
public:
    virtual ~DataItem(void) {}

    virtual std::size_t nSubItems(void) const = 0;
    virtual void        addSubItem(std::shared_ptr<DataItem>&& dataItem) = 0;

    virtual std::shared_ptr<DataItem>  popSubItem(void) = 0;
    virtual DataItem*                  getSubItem(const std::size_t i) = 0;

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

