#ifndef NULL_ITEM_H__
#define NULL_ITEM_H__

#include "DataItem.h"

namespace orchestration {

class NullItem : public DataItem {
public:
    NullItem(void) : DataItem{} { };
    ~NullItem(void)             { };

    std::size_t nSubItems(void) const override
        { throw std::logic_error("[NullItem::nSubItems] No sub items"); };
    void        addSubItem(std::shared_ptr<DataItem>&& dataItem) override
        { throw std::logic_error("[NullItem::addSubItem] No sub items"); };
    std::shared_ptr<DataItem>  popSubItem(void) override
        { throw std::logic_error("[NullItem::popSubItem] No sub items"); };
    DataItem*                  getSubItem(const std::size_t i) override
        { throw std::logic_error("[NullItem::getSubItem] No sub items"); };
};

}

#endif

