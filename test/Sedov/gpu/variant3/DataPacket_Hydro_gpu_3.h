#ifndef DATAPACKET_HYDRO_GPU_3_UNIQUE_IFNDEF_H_
#define DATAPACKET_HYDRO_GPU_3_UNIQUE_IFNDEF_H_

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>
#include <climits>

using real = milhoja::Real;
using milhoja::FArray4D;
using milhoja::IntVect;
using milhoja::RealVect;

class DataPacket_Hydro_gpu_3 : public milhoja::DataPacket {
public:
    // constructor
    DataPacket_Hydro_gpu_3(
    real dt
    
    );
    // destructor
    ~DataPacket_Hydro_gpu_3(void);

    //helper methods from base DataPacket class.
    std::unique_ptr<milhoja::DataPacket> clone(void) const override;
    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3(const DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&& packet) = delete;
    DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3& operator=(const DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&& rhs) = delete;

    // pack and unpack functions from base class.
    void pack(void) override;
    void unpack(void) override;

    // TODO: Streams should be stored inside of an array.

    // DataPacket members are made public so a matching task function can easily access them.
    // Since both files are auto-generated and not maintained by humans, this is fine.
    real _dt_h;
    real* _dt_d;
    std::size_t _nTiles_h;
    std::size_t* _nTiles_d;
    RealVect* _tile_deltas_d;
    IntVect* _tile_lo_d;
    IntVect* _tile_hi_d;
    real* _U_d;
    real* _U_p;
    real* _auxC_d;
    real* _flX_d;
    real* _flY_d;
    real* _flZ_d;
    FArray4D* _f4_U_d;
    FArray4D* _f4_auxC_d;
    FArray4D* _f4_flX_d;
    FArray4D* _f4_flY_d;
    FArray4D* _f4_flZ_d;
    
private:
    static constexpr std::size_t ALIGN_SIZE=16;
    static constexpr std::size_t pad(const std::size_t size) {
        return (((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    }

    // TODO: Streams should be stored inside of an array. Doing so would simplify the code
    // generation & source code for the stream functions.

    static constexpr std::size_t SIZE_DT = sizeof(real);
    static constexpr std::size_t SIZE_NTILES = sizeof(std::size_t);
    static constexpr std::size_t SIZE_FARRAY4D = sizeof(FArray4D);
    static constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
    static constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
    static constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
    static constexpr std::size_t SIZE_U = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (8 - 0 + 1) * sizeof(real);
    static constexpr std::size_t SIZE_AUXC = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * sizeof(real);
    static constexpr std::size_t SIZE_FLX = ((16 + 2 * 0) + 1) * (16 + 2 * 0) * (1 + 2 * 0) * (5) * sizeof(real);
    static constexpr std::size_t SIZE_FLY = (16 + 2 * 0) * ((16 + 2 * 0) + 1) * (1 + 2 * 0) * (5) * sizeof(real);
    static constexpr std::size_t SIZE_FLZ = (1) * (1) * (1) * (1) * sizeof(real);
    
};

#endif
