#ifndef DATA_PACKET_GPU_1_STREAM_H__
#define DATA_PACKET_GPU_1_STREAM_H__

#include "DataPacket.h"

namespace orchestration {

class DataPacket_gpu_1_stream : public DataPacket {
public:
    std::unique_ptr<DataPacket>  clone(void) const override;

    DataPacket_gpu_1_stream(void)  { };
    ~DataPacket_gpu_1_stream(void) { };

    DataPacket_gpu_1_stream(DataPacket_gpu_1_stream&)                  = delete;
    DataPacket_gpu_1_stream(const DataPacket_gpu_1_stream&)            = delete;
    DataPacket_gpu_1_stream(DataPacket_gpu_1_stream&& packet)          = delete;
    DataPacket_gpu_1_stream& operator=(DataPacket_gpu_1_stream&)       = delete;
    DataPacket_gpu_1_stream& operator=(const DataPacket_gpu_1_stream&) = delete;
    DataPacket_gpu_1_stream& operator=(DataPacket_gpu_1_stream&& rhs)  = delete;

    void    pack(void) override;
};

}

#endif

