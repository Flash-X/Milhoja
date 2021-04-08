#ifndef DATA_PACKET_GPU_1_H__
#define DATA_PACKET_GPU_1_H__

#include <stdexcept>

#include "Grid_REAL.h"
#include "DataPacket.h"

namespace orchestration {

class DataPacket_gpu_1 : public DataPacket {
public:
    std::unique_ptr<DataPacket>  clone(void) const override;

    DataPacket_gpu_1(void)  { };
    ~DataPacket_gpu_1(void) { };

    DataPacket_gpu_1(DataPacket_gpu_1&)                  = delete;
    DataPacket_gpu_1(const DataPacket_gpu_1&)            = delete;
    DataPacket_gpu_1(DataPacket_gpu_1&& packet)          = delete;
    DataPacket_gpu_1& operator=(DataPacket_gpu_1&)       = delete;
    DataPacket_gpu_1& operator=(const DataPacket_gpu_1&) = delete;
    DataPacket_gpu_1& operator=(DataPacket_gpu_1&& rhs)  = delete;

    // Overrides of DataPacket member functions
    void    pack(void) override;
    Real*   timeStepGpu(void) const override  { throw std::logic_error("Not implemented"); }

protected:
    // Fix to one block per data packet as first step but with a scratch block
    static constexpr std::size_t    N_BLOCKS = 2; 
};

}

#endif

