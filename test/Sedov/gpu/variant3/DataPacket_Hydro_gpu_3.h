#ifndef DATA_PACKET_HYDRO_GPU_3_H__
#define DATA_PACKET_HYDRO_GPU_3_H__

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h> 

class DataPacket_Hydro_gpu_3 : public milhoja::DataPacket {
public:
    std::unique_ptr<milhoja::DataPacket>  clone(void) const override;

    DataPacket_Hydro_gpu_3(const milhoja::Real new_dt);
    ~DataPacket_Hydro_gpu_3(void);

    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&)                  = delete;
    DataPacket_Hydro_gpu_3(const DataPacket_Hydro_gpu_3&)            = delete;
    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&& packet)          = delete;
    DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&)       = delete;
    DataPacket_Hydro_gpu_3& operator=(const DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&& rhs)  = delete;

    void    pack(void) override;
    void    unpack(void) override;

#if MILHOJA_NDIM == 3 && defined(MILHOJA_OPENACC_OFFLOADING)
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

private:
#if MILHOJA_NDIM == 3
    milhoja::Stream  stream2_;
    milhoja::Stream  stream3_;
#endif
    milhoja::Real*   dt_d_;
    milhoja::Real	 dt=0;

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

