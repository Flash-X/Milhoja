#include "DataPacket_gpu_tf_ener.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_Grid.h>
#include <Milhoja_RuntimeBackend.h>

std::unique_ptr<milhoja::DataPacket> DataPacket_gpu_tf_ener::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new DataPacket_gpu_tf_ener {
            
            
        }
    };
}

// Constructor arguments for DataPacket classes are copied by value into non-reference data members.
// Thus, these values are frozen at instantiation.
DataPacket_gpu_tf_ener::DataPacket_gpu_tf_ener(


)
    :
    milhoja::DataPacket{},
_nTiles_h{0},
_nTiles_d{nullptr},
_tile_deltas_d{nullptr},
_tile_lo_d{nullptr},
_tile_hi_d{nullptr},
_Uin_d{nullptr},
_Uin_p{nullptr},
_Uout_d{nullptr},
_Uout_p{nullptr},
_f4_Uin_d{nullptr},
_f4_Uout_d{nullptr}

    {
}

DataPacket_gpu_tf_ener::~DataPacket_gpu_tf_ener(void) {
}


void DataPacket_gpu_tf_ener::pack(void) {
    using namespace milhoja;
    std::string errMsg = isNull();
    if (errMsg != "")
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] " + errMsg);
    else if (tiles_.size() == 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] No tiles added.");

    // note: cannot set ntiles in the constructor because tiles_ is not filled yet.
    _nTiles_h = tiles_.size();

    constexpr std::size_t SIZE_CONSTRUCTOR = pad(
    SIZE_NTILES
    
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] SIZE_CONSTRUCTOR padding failure");

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * (
    (2 * SIZE_FARRAY4D) + SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI
    
    ));
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] SIZE_TILEMETADATA padding failure");

    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
    SIZE_UIN
    
    ));
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] SIZE_TILEIN padding failure");

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] SIZE_TILEINOUT padding failure");

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
    SIZE_UOUT
    
    ));
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] SIZE_TILEOUT padding failure");

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener pack] SIZE_TILESCRATCH padding failure");

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket, &packet_p_, nBytesPerPacket + SIZE_TILESCRATCH, &packet_d_);

    // pointer determination phase
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);

    
    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ = static_cast<char*>(packet_d_) + SIZE_TILESCRATCH;
    char* ptr_p = copyInStart_p_;
    ptr_d = copyInStart_d_;

    std::size_t* _nTiles_p = static_cast<std::size_t*>( static_cast<void*>(ptr_p) );
    _nTiles_d = static_cast<std::size_t*>( static_cast<void*>(ptr_d) );
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
    
    FArray4D* _f4_Uin_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_Uin_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * SIZE_FARRAY4D;
    ptr_d += _nTiles_h * SIZE_FARRAY4D;
    
    FArray4D* _f4_Uout_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_Uout_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * SIZE_FARRAY4D;
    ptr_d += _nTiles_h * SIZE_FARRAY4D;
    
    
    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    _Uin_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _Uin_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_UIN;
    ptr_d+=_nTiles_h * SIZE_UIN;
    
    
    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;
    
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;
    _Uout_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _Uout_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_UOUT;
    ptr_d+=_nTiles_h * SIZE_UOUT;
    
    
    //memcopy phase
    std::memcpy(_nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);
    
    char* char_ptr;
    for (auto n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[DataPacket_gpu_tf_ener pack] Bad tiledesc.");
        const auto deltas = tileDesc_h->deltas();
        const auto lo = tileDesc_h->lo();
        const auto hi = tileDesc_h->hi();
        
        char_ptr = static_cast<char*>( static_cast<void*>( _tile_deltas_p ) ) + n * SIZE_TILE_DELTAS;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&deltas), SIZE_TILE_DELTAS);
        
        char_ptr = static_cast<char*>( static_cast<void*>( _tile_lo_p ) ) + n * SIZE_TILE_LO;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&lo), SIZE_TILE_LO);
        
        char_ptr = static_cast<char*>( static_cast<void*>( _tile_hi_p ) ) + n * SIZE_TILE_HI;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&hi), SIZE_TILE_HI);
        
        
        real* Uin_d = tileDesc_h->dataPtr();
        constexpr std::size_t offset_Uin = (8 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (1 + 2 * 1 * MILHOJA_K3D) * static_cast<std::size_t>(0);
        constexpr std::size_t nBytes_Uin = (8 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (1 + 2 * 1 * MILHOJA_K3D) * ( 1 - 0 + 1 ) * sizeof(real);
        char_ptr = static_cast<char*>( static_cast<void*>(_Uin_p) ) + n * SIZE_UIN;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(Uin_d + offset_Uin), nBytes_Uin);
        
        FArray4D Uin_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_Uin_d) ) + n * SIZE_UIN)), tileDesc_h->loGC(), tileDesc_h->hiGC(), 1 - 0 + 1};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_Uin_p) ) + n * SIZE_FARRAY4D;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&Uin_device), SIZE_FARRAY4D);
        
        FArray4D Uout_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_Uout_d) ) + n * SIZE_UOUT)), tileDesc_h->loGC(), tileDesc_h->hiGC(), 1 - 0 + 1};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_Uout_p) ) + n * SIZE_FARRAY4D;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&Uout_device), SIZE_FARRAY4D);
        
        
        
        
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[DataPacket_gpu_tf_ener pack] Unable to acquire stream 1.");
}

void DataPacket_gpu_tf_ener::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0)
        throw std::logic_error("[DataPacket_gpu_tf_ener unpack] Empty data packet.");
    if (!stream_.isValid())
        throw std::logic_error("[DataPacket_gpu_tf_ener unpack] Stream not acquired.");
    RuntimeBackend::instance().releaseStream(stream_);
    assert(!stream_.isValid());
    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        real* Uout_data_h = tileDesc_h->dataPtr();
        
        real* Uout_data_p = static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _Uout_p ) ) + n * SIZE_UOUT ) );
        
        
        constexpr std::size_t offset_Uout = (8 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (1 + 2 * 1 * MILHOJA_K3D) * static_cast<std::size_t>(1);
        real*        start_h_Uout = Uout_data_h + offset_Uout;
        const real*  start_p_Uout = Uout_data_p + offset_Uout;
        constexpr std::size_t nBytes_Uout = (8 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (1 + 2 * 1 * MILHOJA_K3D) * ( 1 - 1 + 1 ) * sizeof(real);
        std::memcpy(static_cast<void*>(start_h_Uout), static_cast<const void*>(start_p_Uout), nBytes_Uout);
        
        
    }
}
