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
    int _nTiles_h;
    int* _nTiles_d;
    real* _tile_deltas_d;
    int* _tile_lo_d;
    int* _tile_hi_d;
    int* _tile_loGC_d;
    real* _U_d;
    real* _U_p;
    real* _auxC_d;
    real* _FCX_d;
    real* _FCY_d;
    real* _FCZ_d;
    
private:
    static constexpr std::size_t ALIGN_SIZE=16;
    static constexpr std::size_t pad(const std::size_t size) {
        return (((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    }

    // TODO: Streams should be stored inside of an array. Doing so would simplify the code
    // generation & source code for the stream functions.

    static constexpr std::size_t SIZE_DT = sizeof(real);
    static constexpr std::size_t SIZE_NTILES = sizeof(int);
    static constexpr std::size_t SIZE_TILE_DELTAS = MILHOJA_MDIM * sizeof(real);
    static constexpr std::size_t SIZE_TILE_LO = MILHOJA_MDIM * sizeof(int);
    static constexpr std::size_t SIZE_TILE_HI = MILHOJA_MDIM * sizeof(int);
    static constexpr std::size_t SIZE_TILE_LOGC = MILHOJA_MDIM * sizeof(int);
    static constexpr std::size_t SIZE_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * (9 - 0 + 1) * sizeof(real);
    static constexpr std::size_t SIZE_AUXC = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * sizeof(real);
    static constexpr std::size_t SIZE_FCX = ((8 + 2 * 1) + 1) * (8 + 2 * 1) * (1 + 2 * 0) * (5) * sizeof(real);
    static constexpr std::size_t SIZE_FCY = (8 + 2 * 1) * ((8 + 2 * 1) + 1) * (1 + 2 * 0) * (5) * sizeof(real);
    static constexpr std::size_t SIZE_FCZ = (1) * (1) * (1) * (1) * sizeof(real);
    
};

#endif
