#ifndef DATAPACKET_GPU_DE_1_STREAM_UNIQUE_IFNDEF_H_
#define DATAPACKET_GPU_DE_1_STREAM_UNIQUE_IFNDEF_H_

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>
#include <climits>

using real = milhoja::Real;
using milhoja::FArray4D;
using milhoja::IntVect;
using milhoja::RealVect;

class DataPacket_gpu_de_1_stream : public milhoja::DataPacket {
public:
    // constructor
    DataPacket_gpu_de_1_stream(
    
    
    );
    // destructor
    ~DataPacket_gpu_de_1_stream(void);

    //helper methods from base DataPacket class.
    std::unique_ptr<milhoja::DataPacket> clone(void) const override;
    DataPacket_gpu_de_1_stream(DataPacket_gpu_de_1_stream&) = delete;
    DataPacket_gpu_de_1_stream(const DataPacket_gpu_de_1_stream&) = delete;
    DataPacket_gpu_de_1_stream(DataPacket_gpu_de_1_stream&& packet) = delete;
    DataPacket_gpu_de_1_stream& operator=(DataPacket_gpu_de_1_stream&) = delete;
    DataPacket_gpu_de_1_stream& operator=(const DataPacket_gpu_de_1_stream&) = delete;
    DataPacket_gpu_de_1_stream& operator=(DataPacket_gpu_de_1_stream&& rhs) = delete;

    // pack and unpack functions from base class.
    void pack(void) override;
    void unpack(void) override;

    // TODO: Streams should be stored inside of an array.

    // DataPacket members are made public so a matching task function can easily access them.
    // Since both files are auto-generated and not maintained by humans, this is fine.
    std::size_t _nTiles_h;
    std::size_t* _nTiles_d;
    RealVect* _tile_deltas_d;
    IntVect* _tile_lo_d;
    IntVect* _tile_hi_d;
    real* _Uin_d;
    real* _Uin_p;
    real* _Uout_d;
    real* _Uout_p;
    FArray4D* _f4_Uin_d;
    FArray4D* _f4_Uout_d;
    
private:
    static constexpr std::size_t ALIGN_SIZE=16;
    static constexpr std::size_t pad(const std::size_t size) {
        return (((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    }

    // TODO: Streams should be stored inside of an array. Doing so would simplify the code
    // generation & source code for the stream functions.

    static constexpr std::size_t SIZE_NTILES = sizeof(std::size_t);
    static constexpr std::size_t SIZE_FARRAY4D = sizeof(FArray4D);
    static constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
    static constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
    static constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
    static constexpr std::size_t SIZE_UIN = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (1 - 0 + 1) * sizeof(real);
    static constexpr std::size_t SIZE_UOUT = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
    
};

#endif
