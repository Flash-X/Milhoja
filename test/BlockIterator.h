#ifndef BLOCK_ITERATOR_H__
#define BLOCK_ITERATOR_H__

#include "Grid.h"
#include "Block.h"

class BlockIterator {
public:
    BlockIterator(Grid* myGrid);
    ~BlockIterator(void);

    void  clear(void);
    bool  isValid(void) const;
    Block currentBlock(void);
    void  next(void); 

private:
    // Don't allow copying of iterators
    BlockIterator& operator=(const BlockIterator& rhs);
    BlockIterator(const BlockIterator& other);

    Grid*         grid_;
    unsigned int  idx_;
};

#endif

