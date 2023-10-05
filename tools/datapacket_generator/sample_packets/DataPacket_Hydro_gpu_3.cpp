#include "DataPacket_Hydro_gpu_3.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_Grid.h>
#include <Milhoja_RuntimeBackend.h>

std::unique_ptr<milhoja::DataPacket> DataPacket_Hydro_gpu_3::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new DataPacket_Hydro_gpu_3 {
            _dt_h
            
        }
    };
}

// Constructor arguments for DataPacket classes are copied by value into non-reference data members.
// Thus, these values are frozen at instantiation.
DataPacket_Hydro_gpu_3::DataPacket_Hydro_gpu_3(
real dt

)
    :
    milhoja::DataPacket{},
_dt_h{dt},
_dt_d{nullptr},
_nTiles_h{0},
_nTiles_d{nullptr},
_tile_deltas_d{nullptr},
_tile_lo_d{nullptr},
_tile_hi_d{nullptr},
_tile_loGC_d{nullptr},
_U_d{nullptr},
_U_p{nullptr},
_auxC_d{nullptr},
_FCX_d{nullptr},
_FCY_d{nullptr},
_FCZ_d{nullptr}

    {
}

DataPacket_Hydro_gpu_3::~DataPacket_Hydro_gpu_3(void) {
}


void DataPacket_Hydro_gpu_3::pack(void) {
    using namespace milhoja;
    std::string errMsg = isNull();
    if (errMsg != "")
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] " + errMsg);
    else if (tiles_.size() == 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] No tiles added.");

    // note: cannot set ntiles in the constructor because tiles_ is not filled yet.
    // Check for overflow first to avoid UB
    // TODO: Should casting be checked here or in base class?
    if (tiles_.size() > INT_MAX)
    	throw std::overflow_error("[DataPacket_Hydro_gpu_3 pack] nTiles was too large for int.");
    _nTiles_h = static_cast<int>(tiles_.size());

    constexpr std::size_t SIZE_CONSTRUCTOR = pad(
    SIZE_DT + SIZE_NTILES
    
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] SIZE_CONSTRUCTOR padding failure");

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * (
    SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI + SIZE_TILE_LOGC + 0
    
    ));
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] SIZE_TILEMETADATA padding failure");

    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] SIZE_TILEIN padding failure");

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
    SIZE_U
    
    ));
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] SIZE_TILEINOUT padding failure");

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] SIZE_TILEOUT padding failure");

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
    SIZE_AUXC + SIZE_FCX + SIZE_FCY + SIZE_FCZ
    
    ));
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 pack] SIZE_TILESCRATCH padding failure");

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket, &packet_p_, nBytesPerPacket + SIZE_TILESCRATCH, &packet_d_);

    // pointer determination phase
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);

    _auxC_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_AUXC;
    
    _FCX_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_FCX;
    
    _FCY_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_FCY;
    
    _FCZ_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_FCZ;
    
    
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
    real* _tile_deltas_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _tile_deltas_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_DELTAS;
    ptr_d+=_nTiles_h * SIZE_TILE_DELTAS;
    
    int* _tile_lo_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _tile_lo_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_LO;
    ptr_d+=_nTiles_h * SIZE_TILE_LO;
    
    int* _tile_hi_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _tile_hi_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_HI;
    ptr_d+=_nTiles_h * SIZE_TILE_HI;
    
    int* _tile_loGC_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _tile_loGC_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_LOGC;
    ptr_d+=_nTiles_h * SIZE_TILE_LOGC;
    
    
    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    
    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;
    _U_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _U_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_U;
    ptr_d+=_nTiles_h * SIZE_U;
    
    
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;
    
    //memcopy phase
    std::memcpy(_dt_p, static_cast<void*>(&_dt_h), SIZE_DT);
    std::memcpy(_nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);
    
    char* char_ptr;
    for (auto n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[DataPacket_Hydro_gpu_3 pack] Bad tiledesc.");
        const auto deltas = tileDesc_h->deltas();
        const auto lo = tileDesc_h->lo();
        const auto hi = tileDesc_h->hi();
        const auto loGC = tileDesc_h->loGC();
        
        real _tile_deltas_h[MILHOJA_MDIM] = { deltas.I(), deltas.J(), deltas.K() };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_deltas_p)) + n * SIZE_TILE_DELTAS;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_deltas_h), SIZE_TILE_DELTAS);
        
        int _tile_lo_h[MILHOJA_MDIM] = { lo.I()+1, lo.J()+1, lo.K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_lo_p)) + n * SIZE_TILE_LO;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_lo_h), SIZE_TILE_LO);
        
        int _tile_hi_h[MILHOJA_MDIM] = { hi.I()+1, hi.J()+1, hi.K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_hi_p)) + n * SIZE_TILE_HI;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_hi_h), SIZE_TILE_HI);
        
        int _tile_loGC_h[MILHOJA_MDIM] = { loGC.I()+1, loGC.J()+1, loGC.K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_loGC_p)) + n * SIZE_TILE_LOGC;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_loGC_h), SIZE_TILE_LOGC);
        
        
        
        real* U_d = tileDesc_h->dataPtr();
        constexpr std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        constexpr std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 9 - 0 + 1 ) * sizeof(real);
        char_ptr = static_cast<char*>( static_cast<void*>(_U_p) ) + n * SIZE_U;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(U_d + offset_U), nBytes_U);
        
        
        
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[DataPacket_Hydro_gpu_3 pack] Unable to acquire stream 1.");
}

void DataPacket_Hydro_gpu_3::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0)
        throw std::logic_error("[DataPacket_Hydro_gpu_3 unpack] Empty data packet.");
    if (!stream_.isValid())
        throw std::logic_error("[DataPacket_Hydro_gpu_3 unpack] Stream not acquired.");
    RuntimeBackend::instance().releaseStream(stream_);
    assert(!stream_.isValid());
    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        real* U_data_h = tileDesc_h->dataPtr();
        
        real* U_data_p = static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _U_p ) ) + n * SIZE_U ) );
        
        constexpr std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        real*        start_h_U = U_data_h + offset_U;
        const real*  start_p_U = U_data_p + offset_U;
        constexpr std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 8 - 0 + 1 ) * sizeof(real);
        std::memcpy(static_cast<void*>(start_h_U), static_cast<const void*>(start_p_U), nBytes_U);
        
        
        
    }
}
