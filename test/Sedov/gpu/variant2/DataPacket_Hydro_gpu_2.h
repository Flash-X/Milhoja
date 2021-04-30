#ifndef DATA_PACKET_HYDRO_GPU_2_H__
#define DATA_PACKET_HYDRO_GPU_2_H__

#include "Grid_REAL.h"
#include "DataPacket.h"

namespace orchestration {

class DataPacket_Hydro_gpu_2 : public DataPacket {
public:
    std::unique_ptr<DataPacket>  clone(void) const override;

    DataPacket_Hydro_gpu_2(void);
    ~DataPacket_Hydro_gpu_2(void);

    DataPacket_Hydro_gpu_2(DataPacket_Hydro_gpu_2&)                  = delete;
    DataPacket_Hydro_gpu_2(const DataPacket_Hydro_gpu_2&)            = delete;
    DataPacket_Hydro_gpu_2(DataPacket_Hydro_gpu_2&& packet)          = delete;
    DataPacket_Hydro_gpu_2& operator=(DataPacket_Hydro_gpu_2&)       = delete;
    DataPacket_Hydro_gpu_2& operator=(const DataPacket_Hydro_gpu_2&) = delete;
    DataPacket_Hydro_gpu_2& operator=(DataPacket_Hydro_gpu_2&& rhs)  = delete;

    void    pack(void) override;

#ifdef ENABLE_OPENACC_OFFLOAD
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

private:
    static constexpr  unsigned int  N_STREAMS = 3;

    Stream  streams_[N_STREAMS];
    Real*   dt_d_;
};

}

#endif

