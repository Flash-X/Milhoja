
#include <iostream>
#include "DataPacket_gpu_dens_ener_stream.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_Grid.h>
#include <Milhoja_RuntimeBackend.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>

#if 0
int _nTiles_h;
int* _nTiles_d;
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

int extraAsynchronousQueue(const unsigned int id) override;
void releaseExtraQueue(const unsigned int id) override;


Stream stream2_;

#endif

std::unique_ptr<milhoja::DataPacket> DataPacket_gpu_dens_ener_stream::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new DataPacket_gpu_dens_ener_stream {
            
            
        }
    };
}

DataPacket_gpu_dens_ener_stream::DataPacket_gpu_dens_ener_stream(


)
    :
    milhoja::DataPacket{},
_nTiles_h{tiles_.size()},
_nTiles_d{nullptr},
_tile_deltas_d{nullptr},
_tile_lo_d{nullptr},
_tile_hi_d{nullptr},
_Uin_d{nullptr},
_Uout_d{nullptr}

    {
}

DataPacket_gpu_dens_ener_stream::~DataPacket_gpu_dens_ener_stream(void) {
    if (stream2_.isValid()) throw std::logic_error("[DataPacket_gpu_dens_ener_stream::~DataPacket_gpu_dens_ener_stream] Stream 2 not released");
    
    nullify();
}

int DataPacket_gpu_dens_ener_stream::extraAsynchronousQueue(const unsigned int id) {
	if((id < 2) || (id > 1 + 1)) throw std::invalid_argument("[DataPacket_gpu_dens_ener_stream::extraAsynchronousQueue] Invalid id.");
	switch(id) {
		case 2: if(!stream2_.isValid()) { throw std::logic_error("[DataPacket_gpu_dens_ener_stream::extraAsynchronousQueue] Stream 2 invalid."); } return stream2_.accAsyncQueue;
	}
	return 0;
}
void DataPacket_gpu_dens_ener_stream::releaseExtraQueue(const unsigned int id) {
	if((id < 2) || (id > 1 + 1)) throw std::invalid_argument("[DataPacket_gpu_dens_ener_stream::releaseExtraQueue] Invalid id.");
	switch(id) {
		case 2: if(!stream2_.isValid()) { throw std::logic_error("[DataPacket_gpu_dens_ener_stream::releaseExtraQueue] Stream 2 invalid."); } milhoja::RuntimeBackend::instance().releaseStream(stream2_); break;
	}
}


void DataPacket_gpu_dens_ener_stream::pack(void) {
    using namespace milhoja;
	std::string errMsg = isNull();
	if (errMsg != "")
		throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] " + errMsg);
	else if (tiles_.size() == 0)
		throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] No tiles added.");
    static_assert(sizeof(char) == 1);

    _nTiles_h = tiles_.size();
    // size determination
    constexpr std::size_t SIZE_NTILES =  pad( sizeof(int) );
    constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
    constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
    constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
    constexpr std::size_t SIZE_UIN = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (1 - 0 + 1) * sizeof(real);
    constexpr std::size_t SIZE_UOUT = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
    

	std::size_t SIZE_CONSTRUCTOR = pad(
    SIZE_NTILES
    
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] SIZE_CONSTRUCTOR padding failure");

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * (
    (2 * sizeof(FArray4D)) + SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI + 0
    
    ));
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] SIZE_TILEMETADATA padding failure");

    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
    SIZE_UIN
    
    ));
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] SIZE_TILEIN padding failure");

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] SIZE_TILEINOUT padding failure");

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
    SIZE_UOUT
    
    ));
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] SIZE_TILEOUT padding failure");

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_dens_ener_stream pack] SIZE_TILESCRATCH padding failure");

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT + SIZE_TILESCRATCH;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - SIZE_TILESCRATCH, &packet_p_, nBytesPerPacket, &packet_d_);

    // pointer determination phase
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);

    
    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ = static_cast<char*>(packet_d_) + SIZE_TILESCRATCH;
    char* ptr_p = copyInStart_p_;
    ptr_d = copyInStart_d_;

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
        if (tileDesc_h == nullptr) throw std::runtime_error("[DataPacket_gpu_dens_ener_stream pack] Bad tiledesc.");
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
        
        
        real* Uin_d = tileDesc_h->dataPtr();
        std::size_t offset_Uin = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        std::size_t nBytes_Uin = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
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
        throw std::runtime_error("[DataPacket_gpu_dens_ener_stream pack] Unable to acquire stream 1.");
    stream2_ = RuntimeBackend::instance().requestStream(true);
    if(!stream2_.isValid()) throw std::runtime_error("[DataPacket_gpu_dens_ener_stream::pack] Unable to acquire second stream");
    
}

void DataPacket_gpu_dens_ener_stream::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0) throw std::logic_error("[DataPacket_gpu_dens_ener_stream unpack] Empty data packet.");
    if (!stream_.isValid()) throw std::logic_error("[DataPacket_gpu_dens_ener_stream unpack] Stream not acquired.");
    RuntimeBackend::instance().releaseStream(stream_);
    assert(!stream_.isValid());
    constexpr std::size_t SIZE_UIN = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * (1 - 0 + 1) * sizeof(real);
    constexpr std::size_t SIZE_UOUT = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
    
    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        real* _data_h = tileDesc_h->dataPtr();
        
        real* Uout_data_p = static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _Uout_p ) ) + n * SIZE_UOUT ) );
        
        
        std::size_t offset_ = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
        real*        start_h_ = _data_h + offset_;
        const real*  start_p_Uout = Uout_data_p + offset_;
        std::size_t nBytes_Uout = (8 + 2 * 1) * (16 + 2 * 1) * (1 + 2 * 0) * ( 1 - 0 + 1 ) * sizeof(real);
        std::memcpy(static_cast<void*>(start_h_), static_cast<const void*>(start_p_Uout), nBytes_Uout);
        
        
    }
}
