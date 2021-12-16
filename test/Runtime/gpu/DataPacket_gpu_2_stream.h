#ifndef DATA_PACKET_GPU_2_STREAM_H__
#define DATA_PACKET_GPU_2_STREAM_H__

#include "milhoja.h"
#include "DataPacket.h"

#include "constants.h"
#include "Flash.h"

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
    void    unpack(void) override;

#ifdef ENABLE_OPENACC_OFFLOAD
    int     extraAsynchronousQueue(const unsigned int id) override;
    void    releaseExtraQueue(const unsigned int id) override;
#endif

private:
    Stream   stream2_;

    static constexpr std::size_t    N_ELEMENTS_PER_CC_PER_VARIABLE =   (NXB + 2 * NGUARD * K1D)
                                                                     * (NYB + 2 * NGUARD * K2D)
                                                                     * (NZB + 2 * NGUARD * K3D);
    static constexpr std::size_t    N_ELEMENTS_PER_CC  = N_ELEMENTS_PER_CC_PER_VARIABLE * NUNKVAR;

    static constexpr std::size_t    DELTA_SIZE_BYTES     =          sizeof(RealVect);
    static constexpr std::size_t    CC_BLOCK_SIZE_BYTES  = N_ELEMENTS_PER_CC
                                                                  * sizeof(Real);
    static constexpr std::size_t    POINT_SIZE_BYTES     =          sizeof(IntVect);
    static constexpr std::size_t    ARRAY4_SIZE_BYTES    =          sizeof(FArray4D);
};

}

#endif

