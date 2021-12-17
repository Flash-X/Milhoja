#ifndef DATA_PACKET_GPU_1_STREAM_H__
#define DATA_PACKET_GPU_1_STREAM_H__

#include "DataPacket.h"

namespace orchestration {

class DataPacket_gpu_1_stream : public DataPacket {
public:
    std::unique_ptr<DataPacket>  clone(void) const override;

    DataPacket_gpu_1_stream(void);
    ~DataPacket_gpu_1_stream(void) { };

    DataPacket_gpu_1_stream(DataPacket_gpu_1_stream&)                  = delete;
    DataPacket_gpu_1_stream(const DataPacket_gpu_1_stream&)            = delete;
    DataPacket_gpu_1_stream(DataPacket_gpu_1_stream&& packet)          = delete;
    DataPacket_gpu_1_stream& operator=(DataPacket_gpu_1_stream&)       = delete;
    DataPacket_gpu_1_stream& operator=(const DataPacket_gpu_1_stream&) = delete;
    DataPacket_gpu_1_stream& operator=(DataPacket_gpu_1_stream&& rhs)  = delete;

    void    pack(void) override;
    void    unpack(void) override;

    std::size_t    N_ELEMENTS_PER_CC_PER_VARIABLE;
    std::size_t    N_ELEMENTS_PER_CC;
    std::size_t    DELTA_SIZE_BYTES;
    std::size_t    CC_BLOCK_SIZE_BYTES;
    std::size_t    POINT_SIZE_BYTES;
    std::size_t    ARRAY4_SIZE_BYTES;
};

}

#endif

