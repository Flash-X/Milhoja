#ifndef MOVER_UNPACKER_H__
#define MOVER_UNPACKER_H__

#include "RuntimeElement.h"

namespace orchestration {

class MoverUnpacker : public RuntimeElement {
public:
    MoverUnpacker(void)    { };
    ~MoverUnpacker(void)   { };

    MoverUnpacker(MoverUnpacker&)                  = delete;
    MoverUnpacker(const MoverUnpacker&)            = delete;
    MoverUnpacker(MoverUnpacker&&)                 = delete;
    MoverUnpacker& operator=(MoverUnpacker&)       = delete;
    MoverUnpacker& operator=(const MoverUnpacker&) = delete;
    MoverUnpacker& operator=(MoverUnpacker&&)      = delete;

    void increaseThreadCount(const unsigned int nThreads) override;

    void enqueue(std::shared_ptr<DataItem>&& dataItem) override;
    void closeQueue(void) override;
};

}

#endif

