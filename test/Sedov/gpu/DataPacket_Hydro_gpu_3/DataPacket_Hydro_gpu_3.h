#ifndef DATAPACKET_HYDRO_GPU_3_UNIQUE_IFNDEF_H_
#define DATAPACKET_HYDRO_GPU_3_UNIQUE_IFNDEF_H_

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>

using milhoja::Real;
using milhoja::FArray4D;
using milhoja::Stream;
using milhoja::IntVect;
using milhoja::RealVect;

class DataPacket_Hydro_gpu_3 : public milhoja::DataPacket {
// class DataPacket_Hydro_gpu_3 {
public:
    //constructor / destructor
    DataPacket_Hydro_gpu_3(
    Real dt
    
    );
    ~DataPacket_Hydro_gpu_3(void);

    //helper methods
    //this should eventually be milhoja::DataPacket.
    std::unique_ptr<milhoja::DataPacket> clone(void) const;
    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3(const DataPacket_Hydro_gpu_3&) = delete;
    DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&& packet) = delete;
    DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&)       = delete;
	DataPacket_Hydro_gpu_3& operator=(const DataPacket_Hydro_gpu_3&) = delete;
	DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&& rhs)  = delete;

    void pack(void);
    void unpack(void);


    Real _dt_h;
    Real* _dt_d;
    int _nTiles_h;
    int* _nTiles_d;
    RealVect* _deltas_d;
    IntVect* _lo_d;
    IntVect* _hi_d;
    Real* _U_d;
    Real* _auxC_d;
    Real* _flX_d;
    Real* _flY_d;
    Real* _flZ_d;
    FArray4D* _f4_U_d;
    FArray4D* _f4_auxC_d;
    FArray4D* _f4_flX_d;
    FArray4D* _f4_flY_d;
    FArray4D* _f4_flZ_d;

    Real* _dt_p;
    int* _nTiles_p;
    RealVect* _deltas_p;
    IntVect* _lo_p;
    IntVect* _hi_p;
    Real* _U_p;
    Real* _auxC_p;
    Real* _flX_p;
    Real* _flY_p;
    Real* _flZ_p;
    FArray4D* _f4_U_p;
    FArray4D* _f4_auxC_p;
    FArray4D* _f4_flX_p;
    FArray4D* _f4_flY_p;
    FArray4D* _f4_flZ_p;

    

private:
    static constexpr std::size_t ALIGN_SIZE=16;
    static constexpr std::size_t pad(const std::size_t size) {
        return (((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    }

};

#endif
