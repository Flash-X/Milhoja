#ifndef CUDA_STREAM_H__
#define CUDA_STREAM_H__

#include <driver_types.h>

namespace orchestration {

struct CudaStream {
    static constexpr unsigned int NULL_STREAM_ID = 0;

    unsigned int   id;
    cudaStream_t*  object;
    // TODO: I presently use a pointer because I know that this is compatible
    // with the use of only move semantics in CudaStreamManager.  Study if there
    // is a way to make this a reference and still use move.

    CudaStream(void) : id{NULL_STREAM_ID}, object{nullptr}       { };
    CudaStream(const unsigned int id_in, cudaStream_t* object_in)
        : id{id_in},
          object{object_in}                                      { };
    ~CudaStream(void)                                            { };

    CudaStream(CudaStream&& stream) 
        : id{stream.id},
          object{stream.object}
    {
        stream.id     = NULL_STREAM_ID; 
        stream.object = nullptr;
    }

    CudaStream& operator=(CudaStream&& rhs) {
        id     = rhs.id;
        object = rhs.object;

        rhs.id     = NULL_STREAM_ID;
        rhs.object = nullptr;

        return *this;
    }

    CudaStream(CudaStream&) = delete;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

}
#endif

