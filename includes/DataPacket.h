#ifndef DATA_PACKET_H__
#define DATA_PACKET_H__

#include <deque>
#include <memory>

#include "DataItem.h"

namespace orchestration {

// FIXME: This needs to be written as a full implementation and cleanly.
// TODO:  Should we preset the size of the deque to avoid automatic 
//        resizing when the runtime is active?
class DataPacket : public DataItem {
public:
    DataPacket(void) : DataItem{} { clear(); }
    ~DataPacket(void)             { clear(); }

    std::size_t                nSubItems(void) const override
            { return subItems_.size(); };
    void                       addSubItem(std::shared_ptr<DataItem>&& dataItem) override
            { subItems_.push_front( std::move(dataItem) ); };
    std::shared_ptr<DataItem>  popSubItem(void) override
            { std::shared_ptr<DataItem> item{ std::move(subItems_.front()) };
              subItems_.pop_front();
              return item; };
    DataItem*                  getSubItem(const std::size_t i) override
            { return subItems_[i].get(); };

    // TODO: Do we need to do something elementwise to make certain that 
    //       we are managing memory correctly?
    void clear(void) 
            { id_ = -1; std::deque<std::shared_ptr<DataItem>>().swap(this->subItems_); }

private:
    DataPacket(DataPacket&) = delete;
    DataPacket(const DataPacket&) = delete;
    DataPacket(DataPacket&&) = delete;
    DataPacket& operator=(DataPacket&) = delete;
    DataPacket& operator=(const DataPacket&) = delete;
    DataPacket& operator=(DataPacket&&) = delete;

    int                                     id_ = -1;
    std::deque<std::shared_ptr<DataItem>>   subItems_;
};

}

#endif

