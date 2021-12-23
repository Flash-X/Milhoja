#ifndef DATA_PACKET_HYDRO_GPU_1_H__
#define DATA_PACKET_HYDRO_GPU_1_H__

#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>

class DataPacket_Hydro_gpu_1 : public milhoja::DataPacket {
public:
    std::unique_ptr<milhoja::DataPacket>  clone(void) const override;

    DataPacket_Hydro_gpu_1(void);
    ~DataPacket_Hydro_gpu_1(void);

    DataPacket_Hydro_gpu_1(DataPacket_Hydro_gpu_1&)                  = delete;
    DataPacket_Hydro_gpu_1(const DataPacket_Hydro_gpu_1&)            = delete;
    DataPacket_Hydro_gpu_1(DataPacket_Hydro_gpu_1&& packet)          = delete;
    DataPacket_Hydro_gpu_1& operator=(DataPacket_Hydro_gpu_1&)       = delete;
    DataPacket_Hydro_gpu_1& operator=(const DataPacket_Hydro_gpu_1&) = delete;
    DataPacket_Hydro_gpu_1& operator=(DataPacket_Hydro_gpu_1&& rhs)  = delete;

    void    pack(void) override;
    void    unpack(void) override;

#if ((MILHOJA_NDIM == 2) || (MILHOJA_NDIM == 3)) && defined(MILHOJA_ENABLE_OPENACC_OFFLOAD)
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

private:
#if (MILHOJA_NDIM == 2) || (MILHOJA_NDIM == 3)
    milhoja::Stream  stream2_;
#endif
#if MILHOJA_NDIM == 3
    milhoja::Stream  stream3_;
#endif
    milhoja::Real*   dt_d_;

    std::size_t    N_ELEMENTS_PER_CC_PER_VARIABLE;
    std::size_t    N_ELEMENTS_PER_FCX_PER_VARIABLE;
    std::size_t    N_ELEMENTS_PER_FCX;
    std::size_t    N_ELEMENTS_PER_FCY_PER_VARIABLE;
    std::size_t    N_ELEMENTS_PER_FCY;
    std::size_t    N_ELEMENTS_PER_FCZ_PER_VARIABLE;
    std::size_t    N_ELEMENTS_PER_FCZ;
    std::size_t    DRIVER_DT_SIZE_BYTES;
    std::size_t    DELTA_SIZE_BYTES;
    std::size_t    FCX_BLOCK_SIZE_BYTES;
    std::size_t    FCY_BLOCK_SIZE_BYTES;
    std::size_t    FCZ_BLOCK_SIZE_BYTES;
    std::size_t    POINT_SIZE_BYTES;
    std::size_t    ARRAY4_SIZE_BYTES;
};

#endif

