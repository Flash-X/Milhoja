#ifndef NULL_ITEM_H__
#define NULL_ITEM_H__

#include <Milhoja_DataItem.h>

class NullItem : public milhoja::DataItem {
public:
    NullItem(void) : milhoja::DataItem{} { };
    ~NullItem(void)                      { };
};

#endif

