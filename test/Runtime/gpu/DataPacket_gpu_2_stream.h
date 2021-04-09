#ifndef DATA_PACKET_GPU_2_STREAM_H__
#define DATA_PACKET_GPU_2_STREAM_H__

#include <stdexcept>

#include "Grid_REAL.h"
#include "DataPacket.h"

namespace orchestration {

class DataPacket_gpu_2_stream : public DataPacket {
public:
    std::unique_ptr<DataPacket>  clone(void) const override;

    DataPacket_gpu_2_stream(void) : stream2_() { };
    ~DataPacket_gpu_2_stream(void);

    DataPacket_gpu_2_stream(DataPacket_gpu_2_stream&)                  = delete;
    DataPacket_gpu_2_stream(const DataPacket_gpu_2_stream&)            = delete;
    DataPacket_gpu_2_stream(DataPacket_gpu_2_stream&& packet)          = delete;
    DataPacket_gpu_2_stream& operator=(DataPacket_gpu_2_stream&)       = delete;
    DataPacket_gpu_2_stream& operator=(const DataPacket_gpu_2_stream&) = delete;
    DataPacket_gpu_2_stream& operator=(DataPacket_gpu_2_stream&& rhs)  = delete;

    // Overrides of DataPacket member functions
    void    pack(void) override;

#ifdef ENABLE_OPENACC_OFFLOAD
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

private:
    Stream   stream2_;
};

}

#endif

