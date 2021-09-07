#ifndef DATA_PACKET_HYDRO_GPU_3_H__
#define DATA_PACKET_HYDRO_GPU_3_H__

#include "Grid_REAL.h"
#include "DataPacket.h"

namespace orchestration {

class DataPacket_Hydro_gpu_3 : public DataPacket {
public:
    std::unique_ptr<DataPacket>  clone(void) const override;

    DataPacket_Hydro_gpu_3(void);
    ~DataPacket_Hydro_gpu_3(void);

    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&)                  = delete;
    DataPacket_Hydro_gpu_3(const DataPacket_Hydro_gpu_3&)            = delete;
    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&& packet)          = delete;
    DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&)       = delete;
    DataPacket_Hydro_gpu_3& operator=(const DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&& rhs)  = delete;

    void    pack(void) override;

#if NDIM == 3 && defined(ENABLE_OPENACC_OFFLOAD)
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

    int     nTiles_host(void) const;
    Real    dt_host(void) const;
    void    tileSize_host(int* nxb_h, int* nyb_h, int* nzb_h,
                          int* nvar_h) const;

    int*    nTiles_devptr(void) const;
    Real*   dt_devptr(void) const;

    Real*   deltas_devptr(void) const;
    int*    lo_devptr(void) const;
    int*    hi_devptr(void) const;
    int*    loGC_devptr(void) const;
    int*    hiGC_devptr(void) const;
    Real*   U_devptr(void) const;

private:
#if NDIM == 3
    Stream  stream2_;
    Stream  stream3_;
#endif

    int     nTiles_h_;
    void*   nTiles_p_;
    void*   nTiles_d_;

    Real    dt_h_;
    void*   dt_p_;
    void*   dt_d_;

    void*   deltas_start_p_;
    void*   deltas_start_d_;

    void*   lo_start_p_;
    void*   lo_start_d_;
    void*   hi_start_p_;
    void*   hi_start_d_;

    void*   loGC_start_p_;
    void*   loGC_start_d_;
    void*   hiGC_start_p_;
    void*   hiGC_start_d_;

    void*   U_start_p_;
    void*   U_start_d_;
};

}

#endif

