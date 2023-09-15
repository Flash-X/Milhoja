#ifndef DATAPACKET_GPU_ENER_STREAM_UNIQUE_IFNDEF_H_
#define DATAPACKET_GPU_ENER_STREAM_UNIQUE_IFNDEF_H_

#if 0
_nTiles_h{ tiles_.size() },
_nTiles_d{nullptr},
_tile_deltas_d{nullptr},
_tile_lo_d{nullptr},
_tile_hi_d{nullptr},
_Uin_d{nullptr},
_Uin_p{nullptr},
_Uout_d{nullptr},
_f4_Uin_d{nullptr},
_f4_Uout_d{nullptr},
_f4_Uin_p{nullptr},
_f4_Uout_p{nullptr}

constexpr std::size_t SIZE_NTILES = sizeof(std::size_t);
constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
constexpr std::size_t SIZE_UIN = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (1 - 1 + 1) * sizeof(real);
constexpr std::size_t SIZE_UOUT = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 1 + 1 ) * sizeof(real);


std::size_t* _nTiles_p = static_cast<std::size_t*>( static_cast<void*>(ptr_p) );
_nTiles_d = static_cast<std::size_t*>( static_cast<void*>(ptr_d) );
ptr_p+=SIZE_NTILES;
ptr_d+=SIZE_NTILES;


RealVect* _tile_deltas_p = static_cast<RealVect*>( static_cast<void*>(ptr_p) );
_tile_deltas_d = static_cast<RealVect*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_TILE_DELTAS;
ptr_d+=_nTiles_h * SIZE_TILE_DELTAS;

IntVect* _tile_lo_p = static_cast<IntVect*>( static_cast<void*>(ptr_p) );
_tile_lo_d = static_cast<IntVect*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_TILE_LO;
ptr_d+=_nTiles_h * SIZE_TILE_LO;

IntVect* _tile_hi_p = static_cast<IntVect*>( static_cast<void*>(ptr_p) );
_tile_hi_d = static_cast<IntVect*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_TILE_HI;
ptr_d+=_nTiles_h * SIZE_TILE_HI;

_f4_Uin_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
_f4_Uin_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
ptr_p += _nTiles_h * sizeof(FArray4D);
ptr_d += _nTiles_h * sizeof(FArray4D);

_f4_Uout_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
_f4_Uout_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
ptr_p += _nTiles_h * sizeof(FArray4D);
ptr_d += _nTiles_h * sizeof(FArray4D);


_Uin_p = static_cast<real*>( static_cast<void*>(ptr_p) );
_Uin_d = static_cast<real*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_UIN;
ptr_d+=_nTiles_h * SIZE_UIN;



_Uout_p = static_cast<real*>( static_cast<void*>(ptr_p) );
_Uout_d = static_cast<real*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_UOUT;
ptr_d+=_nTiles_h * SIZE_UOUT;


std::memcpy(_nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);

const auto deltas = tileDesc_h->deltas();
const auto lo = tileDesc_h->lo();
const auto hi = tileDesc_h->hi();
const auto hiGC = tileDesc_h->hiGC();
const auto loGC = tileDesc_h->loGC();

char_ptr = static_cast<char*>( static_cast<void*>( _tile_deltas_p ) ) + n * SIZE_TILE_DELTAS;
std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&deltas), SIZE_TILE_DELTAS);

char_ptr = static_cast<char*>( static_cast<void*>( _tile_lo_p ) ) + n * SIZE_TILE_LO;
std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&lo), SIZE_TILE_LO);

char_ptr = static_cast<char*>( static_cast<void*>( _tile_hi_p ) ) + n * SIZE_TILE_HI;
std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&hi), SIZE_TILE_HI);


real* Uin_d = tileDesc_h->dataPtr();
constexpr std::size_t offset_Uin = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(1);
constexpr std::size_t nBytes_Uin = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 1 + 1 ) * sizeof(real);
char_ptr = static_cast<char*>( static_cast<void*>(_Uin_p) ) + n * SIZE_UIN;
std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(Uin_d + offset_Uin), nBytes_Uin);

FArray4D Uin_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_Uin_d) ) + n * SIZE_UIN)), loGC, hiGC, 1 - 1 + 1};
char_ptr = static_cast<char*>( static_cast<void*>(_f4_Uin_p) ) + n * sizeof(FArray4D);
std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&Uin_device), sizeof(FArray4D));

FArray4D Uout_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_Uout_d) ) + n * SIZE_UOUT)), loGC, hiGC, 1 - 1 + 1};
char_ptr = static_cast<char*>( static_cast<void*>(_f4_Uout_p) ) + n * sizeof(FArray4D);
std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&Uout_device), sizeof(FArray4D));






constexpr std::size_t offset_ = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(1);
real*        start_h_ = _data_h + offset_;
const real*  start_p_Uout = Uout_data_p + offset_;
constexpr std::size_t nBytes_Uout = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 1 + 1 ) * sizeof(real);
std::memcpy(static_cast<void*>(start_h_), static_cast<const void*>(start_p_Uout), nBytes_Uout);


SIZE_NTILES

(2 * sizeof(FArray4D)) + SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI + 0

SIZE_UIN

0

SIZE_UOUT



0

real* _data_h = tileDesc_h->dataPtr();

real* Uout_data_p = static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _Uout_p ) ) + n * SIZE_UOUT ) );

constexpr std::size_t SIZE_UIN = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (1 - 1 + 1) * sizeof(real);
constexpr std::size_t SIZE_UOUT = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 1 + 1 ) * sizeof(real);

#endif

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Stream.h>

using real = milhoja::Real;
using milhoja::FArray4D;
using milhoja::Stream;
using milhoja::IntVect;
using milhoja::RealVect;

class DataPacket_gpu_ener_stream : public milhoja::DataPacket {
public:
    // constructor
    DataPacket_gpu_ener_stream(
    
    
    );
    // destructor
    ~DataPacket_gpu_ener_stream(void);

    //helper methods from base DataPacket class.
    std::unique_ptr<milhoja::DataPacket> clone(void) const override;
    DataPacket_gpu_ener_stream(DataPacket_gpu_ener_stream&) = delete;
    DataPacket_gpu_ener_stream(const DataPacket_gpu_ener_stream&) = delete;
    DataPacket_gpu_ener_stream(DataPacket_gpu_ener_stream&& packet) = delete;
    DataPacket_gpu_ener_stream& operator=(DataPacket_gpu_ener_stream&)       = delete;
	DataPacket_gpu_ener_stream& operator=(const DataPacket_gpu_ener_stream&) = delete;
	DataPacket_gpu_ener_stream& operator=(DataPacket_gpu_ener_stream&& rhs)  = delete;

    // pack and unpack functions from base class.
    void pack(void) override;
    void unpack(void) override;

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
    FArray4D* _f4_Uin_p;
    FArray4D* _f4_Uout_p;
    
private:
    static constexpr std::size_t ALIGN_SIZE=16;
    static constexpr std::size_t pad(const std::size_t size) {
        return (((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    }

};

#endif
