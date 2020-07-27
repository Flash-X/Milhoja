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
};

}

#endif

