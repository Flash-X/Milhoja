
#include <iostream>
#include "DataPacket_Hydro_gpu_1.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_Grid.h>
#include <Milhoja_RuntimeBackend.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>

#if 0
real _dt_h;
real* _dt_d;
int _nTiles_h;
int* _nTiles_d;
RealVect* _tile_deltas_d;
IntVect* _tile_lo_d;
IntVect* _tile_hi_d;
real* _U_d;
real* _U_p;
real* _auxC_d;
real* _auxC_p;
real* _flX_d;
real* _flY_d;
real* _flZ_d;
FArray4D* _f4_U_d;
FArray4D* _f4_auxC_d;
FArray4D* _f4_flX_d;
FArray4D* _f4_flY_d;
FArray4D* _f4_flZ_d;
FArray4D* _f4_U_p;
FArray4D* _f4_auxC_p;
FArray4D* _f4_flX_p;
FArray4D* _f4_flY_p;
FArray4D* _f4_flZ_p;

int extraAsynchronousQueue(const unsigned int id) override;
void releaseExtraQueue(const unsigned int id) override;


Stream stream2_;

#endif

std::unique_ptr<milhoja::DataPacket> DataPacket_Hydro_gpu_1::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new DataPacket_Hydro_gpu_1 {
            _dt_h
            
        }
    };
}

DataPacket_Hydro_gpu_1::DataPacket_Hydro_gpu_1(
real dt

)
    :
    milhoja::DataPacket{},
_dt_h{dt},
_dt_d{nullptr},
_nTiles_h{tiles_.size()},
_nTiles_d{nullptr},
_tile_deltas_d{nullptr},
_tile_lo_d{nullptr},
_tile_hi_d{nullptr},
_U_d{nullptr},
_auxC_d{nullptr},
_flX_d{nullptr},
_flY_d{nullptr},
_flZ_d{nullptr}

    {
}

DataPacket_Hydro_gpu_1::~DataPacket_Hydro_gpu_1(void) {
    if (stream2_.isValid()) throw std::logic_error("[DataPacket_Hydro_gpu_1::~DataPacket_Hydro_gpu_1] Stream 2 not released");
    
    nullify();
}

int DataPacket_Hydro_gpu_1::extraAsynchronousQueue(const unsigned int id) {
	if((id < 2) || (id > 1 + 1)) throw std::invalid_argument("[DataPacket_Hydro_gpu_1::extraAsynchronousQueue] Invalid id.");
	switch(id) {
		case 2: if(!stream2_.isValid()) { throw std::logic_error("[DataPacket_Hydro_gpu_1::extraAsynchronousQueue] Stream 2 invalid."); } return stream2_.accAsyncQueue;
	}
	return 0;
}
void DataPacket_Hydro_gpu_1::releaseExtraQueue(const unsigned int id) {
	if((id < 2) || (id > 1 + 1)) throw std::invalid_argument("[DataPacket_Hydro_gpu_1::releaseExtraQueue] Invalid id.");
	switch(id) {
		case 2: if(!stream2_.isValid()) { throw std::logic_error("[DataPacket_Hydro_gpu_1::releaseExtraQueue] Stream 2 invalid."); } milhoja::RuntimeBackend::instance().releaseStream(stream2_); break;
	}
}


void DataPacket_Hydro_gpu_1::pack(void) {
    using namespace milhoja;
	std::string errMsg = isNull();
	if (errMsg != "")
		throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] " + errMsg);
	else if (tiles_.size() == 0)
		throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] No tiles added.");
    static_assert(sizeof(char) == 1);

    _nTiles_h = tiles_.size();
    // size determination
    constexpr std::size_t SIZE_DT =  pad( sizeof(real));
    constexpr std::size_t SIZE_NTILES =  pad( sizeof(int));
    constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
    constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
    constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
    constexpr std::size_t SIZE_U = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (8 - 0 + 1) * sizeof(real);
    constexpr std::size_t SIZE_AUXC = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 7 - 0 + 1 ) * sizeof(real);
    constexpr std::size_t SIZE_FLX = ((16 + 2 * 0) + 1) * (16 + 2 * 0) * (1 + 2 * 0) * (5) * sizeof(real);
    constexpr std::size_t SIZE_FLY = (16 + 2 * 0) * ((16 + 2 * 0) + 1) * (1 + 2 * 0) * (5) * sizeof(real);
    constexpr std::size_t SIZE_FLZ = (1) * (1) * (1) * (1) * sizeof(real);
    

	std::size_t SIZE_CONSTRUCTOR = pad(
    SIZE_DT + SIZE_NTILES
    
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] SIZE_CONSTRUCTOR padding failure");

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * (
    (5 * sizeof(FArray4D)) + SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI + 0
    
    ));
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] SIZE_TILEMETADATA padding failure");

    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
    SIZE_U
    
    ));
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] SIZE_TILEIN padding failure");

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] SIZE_TILEINOUT padding failure");

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
    SIZE_AUXC
    
    ));
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] SIZE_TILEOUT padding failure");

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
    SIZE_FLX + SIZE_FLY + SIZE_FLZ
    
    ));
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_1 pack] SIZE_TILESCRATCH padding failure");

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT + SIZE_TILESCRATCH;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - SIZE_TILESCRATCH, &packet_p_, nBytesPerPacket, &packet_d_);

    // pointer determination phase
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);

    _flX_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_FLX;
    
    _flY_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_FLY;
    
    _flZ_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_FLZ;
    
    
    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ = static_cast<char*>(packet_d_) + SIZE_TILESCRATCH;
    char* ptr_p = copyInStart_p_;
    ptr_d = copyInStart_d_;

    real* _dt_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _dt_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=SIZE_DT;
    ptr_d+=SIZE_DT;
    
    int* _nTiles_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _nTiles_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=SIZE_NTILES;
    ptr_d+=SIZE_NTILES;
    
    
    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR;
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
    
    _f4_U_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_U_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * sizeof(FArray4D);
    ptr_d += _nTiles_h * sizeof(FArray4D);
    
    _f4_auxC_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_auxC_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * sizeof(FArray4D);
    ptr_d += _nTiles_h * sizeof(FArray4D);
    
    _f4_flX_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_flX_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * sizeof(FArray4D);
    ptr_d += _nTiles_h * sizeof(FArray4D);
    
    _f4_flY_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_flY_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * sizeof(FArray4D);
    ptr_d += _nTiles_h * sizeof(FArray4D);
    
    _f4_flZ_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_flZ_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * sizeof(FArray4D);
    ptr_d += _nTiles_h * sizeof(FArray4D);
    
    
    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    _U_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _U_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_U;
    ptr_d+=_nTiles_h * SIZE_U;
    
    
    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;
    
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;
    _auxC_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _auxC_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_AUXC;
    ptr_d+=_nTiles_h * SIZE_AUXC;
    
    
    //memcopy phase
    std::memcpy(_dt_p, static_cast<void*>(&_dt_h), SIZE_DT);
    std::memcpy(_nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);
    
    char* char_ptr;
    for (std::size_t n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[DataPacket_Hydro_gpu_1 pack] Bad tiledesc.");
        const auto deltas = tileDesc_h->deltas();
        const auto lo = tileDesc_h->lo();
        const auto hi = tileDesc_h->hi();
        const auto loGC = tileDesc_h->loGC();
        const auto hiGC = tileDesc_h->hiGC();
        
        char_ptr = static_cast<char*>( static_cast<void*>( _tile_deltas_p ) ) + n * SIZE_TILE_DELTAS;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&deltas), SIZE_TILE_DELTAS);
        
        char_ptr = static_cast<char*>( static_cast<void*>( _tile_lo_p ) ) + n * SIZE_TILE_LO;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&lo), SIZE_TILE_LO);
        
        char_ptr = static_cast<char*>( static_cast<void*>( _tile_hi_p ) ) + n * SIZE_TILE_HI;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&hi), SIZE_TILE_HI);
        
        
        real* U_d = tileDesc_h->dataPtr();
        std::size_t offset_U = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        std::size_t nBytes_U = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 8 - 0 + 1 ) * sizeof(real);
        char_ptr = static_cast<char*>( static_cast<void*>(_U_p) ) + n * SIZE_U;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(U_d + offset_U), nBytes_U);
        FArray4D U_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_U_d) ) + n * SIZE_U)), loGC, hiGC, 8 - 0 + 1};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_U_p) ) + n * sizeof(FArray4D);
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&U_device), sizeof(FArray4D));
        
        FArray4D auxC_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_auxC_d) ) + n * SIZE_AUXC)), loGC, hiGC, 1};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_auxC_p) ) + n * sizeof(FArray4D);
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&auxC_device), sizeof(FArray4D));
        
        FArray4D flX_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flX_d) ) + n * SIZE_FLX)), lo, IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }, 5};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_flX_p) ) + n * sizeof(FArray4D);
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flX_device), sizeof(FArray4D));
        
        FArray4D flY_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flY_d) ) + n * SIZE_FLY)), lo, IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }, 5};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_flY_p) ) + n * sizeof(FArray4D);
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flY_device), sizeof(FArray4D));
        
        FArray4D flZ_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flZ_d) ) + n * SIZE_FLZ)), lo, IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }, 1};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_flZ_p) ) + n * sizeof(FArray4D);
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flZ_device), sizeof(FArray4D));
        
        
        
        
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[DataPacket_Hydro_gpu_1 pack] Unable to acquire stream 1.");
    stream2_ = RuntimeBackend::instance().requestStream(true);
    if(!stream2_.isValid()) throw std::runtime_error("[DataPacket_Hydro_gpu_1::pack] Unable to acquire second stream");
    
}

void DataPacket_Hydro_gpu_1::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0) throw std::logic_error("[DataPacket_Hydro_gpu_1 unpack] Empty data packet.");
    if (!stream_.isValid()) throw std::logic_error("[DataPacket_Hydro_gpu_1 unpack] Stream not acquired.");
    RuntimeBackend::instance().releaseStream(stream_);
    assert(!stream_.isValid());
    constexpr std::size_t SIZE_U = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (8 - 0 + 1) * sizeof(real);
    constexpr std::size_t SIZE_AUXC = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 7 - 0 + 1 ) * sizeof(real);
    
    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        real* _data_h = tileDesc_h->dataPtr();
        
        real* auxC_data_p = static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _auxC_p ) ) + n * SIZE_AUXC ) );
        
        
        std::size_t offset_ = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        real*        start_h_ = _data_h + offset_;
        const real*  start_p_auxC = auxC_data_p + offset_;
        std::size_t nBytes_ = (16 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 7 - 0 + 1 ) * sizeof(real);
        std::memcpy(static_cast<void*>(start_h_), static_cast<const void*>(start_p_auxC), nBytes_);
        
    }
}
