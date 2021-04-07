#ifndef DATA_PACKET_HYDRO_GPU_1_H__
#define DATA_PACKET_HYDRO_GPU_1_H__

#include "Grid_REAL.h"
#include "DataPacket.h"

namespace orchestration {

class DataPacket_Hydro_gpu_1 : public DataPacket {
public:
    std::unique_ptr<DataPacket>  clone(void) const override;

    DataPacket_Hydro_gpu_1(void);
    ~DataPacket_Hydro_gpu_1(void);

    DataPacket_Hydro_gpu_1(DataPacket_Hydro_gpu_1&)                  = delete;
    DataPacket_Hydro_gpu_1(const DataPacket_Hydro_gpu_1&)            = delete;
    DataPacket_Hydro_gpu_1(DataPacket_Hydro_gpu_1&& packet)          = delete;
    DataPacket_Hydro_gpu_1& operator=(DataPacket_Hydro_gpu_1&)       = delete;
    DataPacket_Hydro_gpu_1& operator=(const DataPacket_Hydro_gpu_1&) = delete;
    DataPacket_Hydro_gpu_1& operator=(DataPacket_Hydro_gpu_1&& rhs)  = delete;

    // Overrides of DataPacket member functions
    void    pack(void) override;
    Real*   timeStepGpu(void) const override  { return dt_d_; }

private:
    Real*   dt_d_;
};

}

#endif

