#ifndef NULL_ITEM_H__
#define NULL_ITEM_H__

#include "DataItem.h"

namespace orchestration {

class NullItem : public DataItem {
public:
    NullItem(void) : DataItem{} { };
    ~NullItem(void)             { };
};

}

#endif

