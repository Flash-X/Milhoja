#include "cgkit.DataPacket_gpu_de_1_stream.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_Grid.h>
#include <Milhoja_RuntimeBackend.h>

#if 0
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


#endif

std::unique_ptr<milhoja::DataPacket> DataPacket_gpu_de_1_stream::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new DataPacket_gpu_de_1_stream {
            
            
        }
    };
}

// Constructor arguments for DataPacket classes are copied by value into non-reference data members.
// Thus, these values are frozen at instantiation.
DataPacket_gpu_de_1_stream::DataPacket_gpu_de_1_stream(


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
_f4_Uin_d{nullptr},
_f4_Uout_d{nullptr},
_f4_Uin_p{nullptr},
_f4_Uout_p{nullptr}

    {
}

DataPacket_gpu_de_1_stream::~DataPacket_gpu_de_1_stream(void) {
}


void DataPacket_gpu_de_1_stream::pack(void) {
    using namespace milhoja;
	std::string errMsg = isNull();
	if (errMsg != "")
		throw std::logic_error("[DataPacket_gpu_de_1_stream pack] " + errMsg);
	else if (tiles_.size() == 0)
		throw std::logic_error("[DataPacket_gpu_de_1_stream pack] No tiles added.");

    // note: cannot set ntiles in the constructor because tiles_ is not filled yet.
    _nTiles_h = tiles_.size();
    // size determination
    constexpr std::size_t SIZE_NTILES = sizeof(std::size_t);
    constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
    constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
    constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
    constexpr std::size_t SIZE_UIN = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (1 - 0 + 1) * sizeof(real);
    constexpr std::size_t SIZE_UOUT = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
    

	std::size_t SIZE_CONSTRUCTOR = pad(
    SIZE_NTILES
    
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_de_1_stream pack] SIZE_CONSTRUCTOR padding failure");

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * (
    (2 * sizeof(FArray4D)) + SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI + 0
    
    ));
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_de_1_stream pack] SIZE_TILEMETADATA padding failure");

    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
    SIZE_UIN
    
    ));
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_de_1_stream pack] SIZE_TILEIN padding failure");

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_de_1_stream pack] SIZE_TILEINOUT padding failure");

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
    SIZE_UOUT
    
    ));
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_de_1_stream pack] SIZE_TILEOUT padding failure");

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_de_1_stream pack] SIZE_TILESCRATCH padding failure");

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
    
    _f4_Uin_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_Uin_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * sizeof(FArray4D);
    ptr_d += _nTiles_h * sizeof(FArray4D);
    
    _f4_Uout_p = static_cast<FArray4D*>( static_cast<void*>( ptr_p ) );
    _f4_Uout_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
    ptr_p += _nTiles_h * sizeof(FArray4D);
    ptr_d += _nTiles_h * sizeof(FArray4D);
    
    
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
    for (std::size_t n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[DataPacket_gpu_de_1_stream pack] Bad tiledesc.");
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
        constexpr std::size_t offset_Uin = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        constexpr std::size_t nBytes_Uin = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
        char_ptr = static_cast<char*>( static_cast<void*>(_Uin_p) ) + n * SIZE_UIN;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(Uin_d + offset_Uin), nBytes_Uin);
        
        FArray4D Uin_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_Uin_d) ) + n * SIZE_UIN)), loGC, hiGC, 1 - 0 + 1};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_Uin_p) ) + n * sizeof(FArray4D);
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&Uin_device), sizeof(FArray4D));
        
        FArray4D Uout_device{ static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_Uout_d) ) + n * SIZE_UOUT)), loGC, hiGC, 1 - 0 + 1};
        char_ptr = static_cast<char*>( static_cast<void*>(_f4_Uout_p) ) + n * sizeof(FArray4D);
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&Uout_device), sizeof(FArray4D));
        
        
        
        
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[DataPacket_gpu_de_1_stream pack] Unable to acquire stream 1.");
}

void DataPacket_gpu_de_1_stream::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0)
        throw std::logic_error("[DataPacket_gpu_de_1_stream unpack] Empty data packet.");
    if (!stream_.isValid())
        throw std::logic_error("[DataPacket_gpu_de_1_stream unpack] Stream not acquired.");
    RuntimeBackend::instance().releaseStream(stream_);
    assert(!stream_.isValid());
    constexpr std::size_t SIZE_UIN = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (1 - 0 + 1) * sizeof(real);
    constexpr std::size_t SIZE_UOUT = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
    
    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        real* _data_h = tileDesc_h->dataPtr();
        
        real* Uout_data_p = static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _Uout_p ) ) + n * SIZE_UOUT ) );
        
        
        constexpr std::size_t offset_ = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        real*        start_h_ = _data_h + offset_;
        const real*  start_p_Uout = Uout_data_p + offset_;
        constexpr std::size_t nBytes_Uout = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
        std::memcpy(static_cast<void*>(start_h_), static_cast<const void*>(start_p_Uout), nBytes_Uout);
        
        
    }
}