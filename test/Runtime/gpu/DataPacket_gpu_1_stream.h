#ifndef DATA_PACKET_GPU_1_STREAM_H__
#define DATA_PACKET_GPU_1_STREAM_H__

#include <stdexcept>

#include "Grid_REAL.h"
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

    // Overrides of DataPacket member functions
    void    pack(void) override;
    Real*   timeStepGpu(void) const override  { throw std::logic_error("Not implemented"); }

protected:
    // Fix to one block per data packet as first step but with a scratch block
    static constexpr std::size_t    N_BLOCKS = 2; 
};

}

#endif

