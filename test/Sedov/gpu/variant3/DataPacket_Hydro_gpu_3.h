#ifndef DATA_PACKET_HYDRO_GPU_3_H__
#define DATA_PACKET_HYDRO_GPU_3_H__

#include "milhoja.h"
#include "Grid_REAL.h"
#include "DataPacket.h"

#include "constants.h"
#include "Flash.h"

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
    void    unpack(void) override;

#if NDIM == 3 && defined(ENABLE_OPENACC_OFFLOAD)
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

private:
#if NDIM == 3
    Stream  stream2_;
    Stream  stream3_;
#endif
    Real*   dt_d_;

    static constexpr std::size_t    N_ELEMENTS_PER_CC_PER_VARIABLE =   (NXB + 2 * NGUARD * K1D)
                                                                     * (NYB + 2 * NGUARD * K2D)
                                                                     * (NZB + 2 * NGUARD * K3D);

    static constexpr std::size_t    N_ELEMENTS_PER_FCX_PER_VARIABLE = (NXB + 1) * NYB * NZB;
    static constexpr std::size_t    N_ELEMENTS_PER_FCX = N_ELEMENTS_PER_FCX_PER_VARIABLE * NFLUXES;

    static constexpr std::size_t    N_ELEMENTS_PER_FCY_PER_VARIABLE = NXB * (NYB + 1) * NZB;
    static constexpr std::size_t    N_ELEMENTS_PER_FCY = N_ELEMENTS_PER_FCY_PER_VARIABLE * NFLUXES;

    static constexpr std::size_t    N_ELEMENTS_PER_FCZ_PER_VARIABLE = NXB * NYB * (NZB + 1);
    static constexpr std::size_t    N_ELEMENTS_PER_FCZ = N_ELEMENTS_PER_FCZ_PER_VARIABLE * NFLUXES;

    static constexpr std::size_t    DRIVER_DT_SIZE_BYTES =          sizeof(Real);
    static constexpr std::size_t    DELTA_SIZE_BYTES     =          sizeof(RealVect);
    static constexpr std::size_t    FCX_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCX
                                                                  * sizeof(Real);
    static constexpr std::size_t    FCY_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCY
                                                                  * sizeof(Real);
    static constexpr std::size_t    FCZ_BLOCK_SIZE_BYTES = N_ELEMENTS_PER_FCZ
                                                                  * sizeof(Real);
    static constexpr std::size_t    POINT_SIZE_BYTES     =          sizeof(IntVect);
    static constexpr std::size_t    ARRAY4_SIZE_BYTES    =          sizeof(FArray4D);
};

}

#endif

