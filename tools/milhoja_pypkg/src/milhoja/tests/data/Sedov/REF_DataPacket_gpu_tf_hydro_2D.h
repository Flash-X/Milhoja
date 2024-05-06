#ifndef DATAPACKET_GPU_TF_HYDRO_UNIQUE_IFNDEF_H_
#define DATAPACKET_GPU_TF_HYDRO_UNIQUE_IFNDEF_H_

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>
#include <climits>

using real = milhoja::Real;
using milhoja::FArray4D;
using milhoja::IntVect;
using milhoja::RealVect;

class DataPacket_gpu_tf_hydro : public milhoja::DataPacket {
public:
    // constructor
    DataPacket_gpu_tf_hydro(
    real external_hydro_op1_dt
    );
    // destructor
    ~DataPacket_gpu_tf_hydro(void);

    //helper methods from base DataPacket class.
    std::unique_ptr<milhoja::DataPacket> clone(void) const override;
    DataPacket_gpu_tf_hydro(DataPacket_gpu_tf_hydro&) = delete;
    DataPacket_gpu_tf_hydro(const DataPacket_gpu_tf_hydro&) = delete;
    DataPacket_gpu_tf_hydro(DataPacket_gpu_tf_hydro&& packet) = delete;
    DataPacket_gpu_tf_hydro& operator=(DataPacket_gpu_tf_hydro&) = delete;
    DataPacket_gpu_tf_hydro& operator=(const DataPacket_gpu_tf_hydro&) = delete;
    DataPacket_gpu_tf_hydro& operator=(DataPacket_gpu_tf_hydro&& rhs) = delete;

    // pack and unpack functions from base class.
    void pack(void) override;
    void unpack(void) override;

    // TODO: Streams should be stored inside of an array.

    // DataPacket members are made public so a matching task function can easily access them.
    // Since both files are auto-generated and not maintained by humans, this is fine.
    std::size_t _nTiles_h;
    std::size_t* _nTiles_d;
    real _external_hydro_op1_dt_h;
    real* _external_hydro_op1_dt_d;
    RealVect* _tile_deltas_d;
    IntVect* _tile_hi_d;
    IntVect* _tile_lo_d;
    real* _CC_1_d;
    real* _CC_1_p;
    real* _scratch_hydro_op1_auxC_d;
    real* _scratch_hydro_op1_flX_d;
    real* _scratch_hydro_op1_flY_d;
    real* _scratch_hydro_op1_flZ_d;
    FArray4D* _f4_CC_1_d;
    FArray4D* _f4_scratch_hydro_op1_auxC_d;
    FArray4D* _f4_scratch_hydro_op1_flX_d;
    FArray4D* _f4_scratch_hydro_op1_flY_d;
    FArray4D* _f4_scratch_hydro_op1_flZ_d;
private:
    static constexpr std::size_t ALIGN_SIZE=16;
    static constexpr std::size_t pad(const std::size_t size) {
        return (((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    }

    // TODO: Streams should be stored inside of an array. Doing so would simplify the code
    // generation & source code for the stream functions.

    static constexpr std::size_t SIZE_NTILES = sizeof(std::size_t);
    static constexpr std::size_t SIZE_EXTERNAL_HYDRO_OP1_DT = sizeof(real);
    static constexpr std::size_t SIZE_FARRAY4D = sizeof(FArray4D);
    static constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
    static constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
    static constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
    static constexpr std::size_t SIZE_CC_1 = (16 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (1 + 2 * 1 * MILHOJA_K3D) * (8 + 1 - 0) * sizeof(real);
    static constexpr std::size_t SIZE_SCRATCH_HYDRO_OP1_AUXC = (18) * (18) * (1) * (1) * sizeof(real);
    static constexpr std::size_t SIZE_SCRATCH_HYDRO_OP1_FLX = (17) * (16) * (1) * (5) * sizeof(real);
    static constexpr std::size_t SIZE_SCRATCH_HYDRO_OP1_FLY = (16) * (17) * (1) * (5) * sizeof(real);
    static constexpr std::size_t SIZE_SCRATCH_HYDRO_OP1_FLZ = (1) * (1) * (1) * (1) * sizeof(real);
};

#endif

