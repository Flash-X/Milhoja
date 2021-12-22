#ifndef DATA_PACKET_HYDRO_GPU_2_H__
#define DATA_PACKET_HYDRO_GPU_2_H__

#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>

class DataPacket_Hydro_gpu_2 : public milhoja::DataPacket {
public:
    std::unique_ptr<milhoja::DataPacket>  clone(void) const override;

    DataPacket_Hydro_gpu_2(void);
    ~DataPacket_Hydro_gpu_2(void);

    DataPacket_Hydro_gpu_2(DataPacket_Hydro_gpu_2&)                  = delete;
    DataPacket_Hydro_gpu_2(const DataPacket_Hydro_gpu_2&)            = delete;
    DataPacket_Hydro_gpu_2(DataPacket_Hydro_gpu_2&& packet)          = delete;
    DataPacket_Hydro_gpu_2& operator=(DataPacket_Hydro_gpu_2&)       = delete;
    DataPacket_Hydro_gpu_2& operator=(const DataPacket_Hydro_gpu_2&) = delete;
    DataPacket_Hydro_gpu_2& operator=(DataPacket_Hydro_gpu_2&& rhs)  = delete;

    void    pack(void) override;
    void    unpack(void) override;

#ifdef ENABLE_OPENACC_OFFLOAD
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

private:
    static constexpr  unsigned int  N_STREAMS = 3;

    milhoja::Stream  streams_[N_STREAMS];
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

