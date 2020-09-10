#ifndef CUDA_MOVER_UNPACKER_H__
#define CUDA_MOVER_UNPACKER_H__

#include "RuntimeElement.h"

namespace orchestration {

class CudaMoverUnpacker : public RuntimeElement {
public:
    CudaMoverUnpacker(void)    { };
    ~CudaMoverUnpacker(void)   { };

    void increaseThreadCount(const unsigned int nThreads) override;

    void enqueue(std::shared_ptr<DataItem>&& dataItem) override;
    void closeQueue(void) override;

private:
    CudaMoverUnpacker(CudaMoverUnpacker&)                  = delete;
    CudaMoverUnpacker(const CudaMoverUnpacker&)            = delete;
    CudaMoverUnpacker(CudaMoverUnpacker&&)                 = delete;
    CudaMoverUnpacker& operator=(CudaMoverUnpacker&)       = delete;
    CudaMoverUnpacker& operator=(const CudaMoverUnpacker&) = delete;
    CudaMoverUnpacker& operator=(CudaMoverUnpacker&&)      = delete;
};

}

#endif

