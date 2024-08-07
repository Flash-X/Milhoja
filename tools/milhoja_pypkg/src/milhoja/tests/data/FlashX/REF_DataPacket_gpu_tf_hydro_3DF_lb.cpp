#include "DataPacket_gpu_tf_hydro_f_lb.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_RuntimeBackend.h>

std::unique_ptr<milhoja::DataPacket> DataPacket_gpu_tf_hydro::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new DataPacket_gpu_tf_hydro {
            _external_hydro_op1_dt_h
            
        }
    };
}

// Constructor arguments for DataPacket classes are copied by value into non-reference data members.
// Thus, these values are frozen at instantiation.
DataPacket_gpu_tf_hydro::DataPacket_gpu_tf_hydro(
real external_hydro_op1_dt

)
    :
    milhoja::DataPacket{},
_external_hydro_op1_dt_h{external_hydro_op1_dt},
_external_hydro_op1_dt_d{nullptr},
_nTiles_h{0},
_nTiles_d{nullptr},
_tile_deltas_d{nullptr},
_tile_lo_d{nullptr},
_tile_hi_d{nullptr},
_tile_interior_d{nullptr},
_tile_lbound_d{nullptr},
_tile_ubound_d{nullptr},
_tile_arrayBounds_d{nullptr},
_lbdd_CC_1_d{nullptr},
_lbdd_scratch_hydro_op1_auxC_d{nullptr},
_CC_1_d{nullptr},
_CC_1_p{nullptr},
_scratch_hydro_op1_auxC_d{nullptr},
_scratch_hydro_op1_flX_d{nullptr},
_scratch_hydro_op1_flY_d{nullptr},
_scratch_hydro_op1_flZ_d{nullptr}

    {
}

DataPacket_gpu_tf_hydro::~DataPacket_gpu_tf_hydro(void) {
    if (stream2_.isValid())
    	throw std::logic_error("[DataPacket_gpu_tf_hydro::~DataPacket_gpu_tf_hydro] Stream 2 not released");
    if (stream3_.isValid())
    	throw std::logic_error("[DataPacket_gpu_tf_hydro::~DataPacket_gpu_tf_hydro] Stream 3 not released");
    
}

int DataPacket_gpu_tf_hydro::extraAsynchronousQueue(const unsigned int id) {
	if (id > INT_MAX)
		throw std::overflow_error("[DataPacket_gpu_tf_hydro extraAsynchronousQueue] id is too large for int.");
	if((id < 2) || (id > 2 + 1))
		throw std::invalid_argument("[DataPacket_gpu_tf_hydro::extraAsynchronousQueue] Invalid id.");
	else if (id == 2) {
		if (!stream2_.isValid())
			throw std::logic_error("[DataPacket_gpu_tf_hydro::extraAsynchronousQueue] Stream 2 invalid.");
		return stream2_.accAsyncQueue;
	} else if (id == 3) {
		if (!stream3_.isValid())
			throw std::logic_error("[DataPacket_gpu_tf_hydro::extraAsynchronousQueue] Stream 3 invalid.");
		return stream3_.accAsyncQueue;
	}
	return 0;
}

void DataPacket_gpu_tf_hydro::releaseExtraQueue(const unsigned int id) {
	if((id < 2) || (id > 2 + 1))
		throw std::invalid_argument("[DataPacket_gpu_tf_hydro::releaseExtraQueue] Invalid id.");
	else if(id == 2) {
		if(!stream2_.isValid())
			throw std::logic_error("[DataPacket_gpu_tf_hydro::releaseExtraQueue] Stream 2 invalid.");
		milhoja::RuntimeBackend::instance().releaseStream(stream2_);
	} else if(id == 3) {
		if(!stream3_.isValid())
			throw std::logic_error("[DataPacket_gpu_tf_hydro::releaseExtraQueue] Stream 3 invalid.");
		milhoja::RuntimeBackend::instance().releaseStream(stream3_);
	}
}


void DataPacket_gpu_tf_hydro::pack(void) {
    using namespace milhoja;
    std::string errMsg = isNull();
    if (errMsg != "")
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] " + errMsg);
    else if (tiles_.size() == 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] No tiles added.");

    // note: cannot set ntiles in the constructor because tiles_ is not filled yet.
    // Check for overflow first to avoid UB
    // TODO: Should casting be checked here or in base class?
    if (tiles_.size() > INT_MAX)
    	throw std::overflow_error("[DataPacket_gpu_tf_hydro pack] nTiles was too large for int.");
    _nTiles_h = static_cast<int>(tiles_.size());

    constexpr std::size_t SIZE_CONSTRUCTOR = pad(
    SIZE_EXTERNAL_HYDRO_OP1_DT + SIZE_NTILES
    
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] SIZE_CONSTRUCTOR padding failure");

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * (
    SIZE_TILE_DELTAS
    + SIZE_TILE_LO
    + SIZE_TILE_HI
    + SIZE_TILE_INTERIOR
    + SIZE_TILE_LBOUND
    + SIZE_TILE_UBOUND
    + SIZE_TILE_ARRAYBOUNDS
    + SIZE_LBDD_CC_1
    + SIZE_LBDD_SCRATCH_HYDRO_OP1_AUXC
    
    ));
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] SIZE_TILEMETADATA padding failure");

    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] SIZE_TILEIN padding failure");

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
    SIZE_CC_1
    
    ));
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] SIZE_TILEINOUT padding failure");

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
    0
    
    ));
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] SIZE_TILEOUT padding failure");

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
    SIZE_SCRATCH_HYDRO_OP1_AUXC + SIZE_SCRATCH_HYDRO_OP1_FLX + SIZE_SCRATCH_HYDRO_OP1_FLY + SIZE_SCRATCH_HYDRO_OP1_FLZ
    
    ));
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro pack] SIZE_TILESCRATCH padding failure");

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket, &packet_p_, nBytesPerPacket + SIZE_TILESCRATCH, &packet_d_);

    // pointer determination phase
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);

    _scratch_hydro_op1_auxC_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_SCRATCH_HYDRO_OP1_AUXC;
    
    _scratch_hydro_op1_flX_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_SCRATCH_HYDRO_OP1_FLX;
    
    _scratch_hydro_op1_flY_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_SCRATCH_HYDRO_OP1_FLY;
    
    _scratch_hydro_op1_flZ_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_d += _nTiles_h * SIZE_SCRATCH_HYDRO_OP1_FLZ;
    
    
    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ = static_cast<char*>(packet_d_) + SIZE_TILESCRATCH;
    char* ptr_p = copyInStart_p_;
    ptr_d = copyInStart_d_;

    real* _external_hydro_op1_dt_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _external_hydro_op1_dt_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=SIZE_EXTERNAL_HYDRO_OP1_DT;
    ptr_d+=SIZE_EXTERNAL_HYDRO_OP1_DT;
    
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

    int* _tile_interior_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _tile_interior_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_INTERIOR;
    ptr_d+=_nTiles_h * SIZE_TILE_INTERIOR;

    int* _tile_lbound_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _tile_lbound_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_LBOUND;
    ptr_d+=_nTiles_h * SIZE_TILE_LBOUND;

    int* _tile_ubound_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _tile_ubound_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_UBOUND;
    ptr_d+=_nTiles_h * SIZE_TILE_UBOUND;

    int* _tile_arrayBounds_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _tile_arrayBounds_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_TILE_ARRAYBOUNDS;
    ptr_d+=_nTiles_h * SIZE_TILE_ARRAYBOUNDS;
    
    int* _lbdd_CC_1_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _lbdd_CC_1_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_LBDD_CC_1;
    ptr_d+=_nTiles_h * SIZE_LBDD_CC_1;

    int* _lbdd_scratch_hydro_op1_auxC_p = static_cast<int*>( static_cast<void*>(ptr_p) );
    _lbdd_scratch_hydro_op1_auxC_d = static_cast<int*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_LBDD_SCRATCH_HYDRO_OP1_AUXC;
    ptr_d+=_nTiles_h * SIZE_LBDD_SCRATCH_HYDRO_OP1_AUXC;

    
    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    
    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;
    _CC_1_p = static_cast<real*>( static_cast<void*>(ptr_p) );
    _CC_1_d = static_cast<real*>( static_cast<void*>(ptr_d) );
    ptr_p+=_nTiles_h * SIZE_CC_1;
    ptr_d+=_nTiles_h * SIZE_CC_1;
    
    
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;
    
    //memcopy phase
    std::memcpy(_external_hydro_op1_dt_p, static_cast<void*>(&_external_hydro_op1_dt_h), SIZE_EXTERNAL_HYDRO_OP1_DT);
    std::memcpy(_nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);
    
    char* char_ptr;
    for (auto n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[DataPacket_gpu_tf_hydro pack] Bad tiledesc.");
        const auto deltas = tileDesc_h->deltas();
        const auto lo = tileDesc_h->lo();
        const auto hi = tileDesc_h->hi();
        const auto lbound = tileDesc_h->loGC();
        const auto ubound = tileDesc_h->hiGC();
        
        real _tile_deltas_h[MILHOJA_MDIM] = { deltas.I(), deltas.J(), deltas.K() };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_deltas_p)) + n * SIZE_TILE_DELTAS;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_deltas_h), SIZE_TILE_DELTAS);
        
        int _tile_lo_h[MILHOJA_MDIM] = { lo.I()+1, lo.J()+1, lo.K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_lo_p)) + n * SIZE_TILE_LO;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_lo_h), SIZE_TILE_LO);
        
        int _tile_hi_h[MILHOJA_MDIM] = { hi.I()+1, hi.J()+1, hi.K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_hi_p)) + n * SIZE_TILE_HI;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_hi_h), SIZE_TILE_HI);

        int _tile_interior_h[MILHOJA_MDIM * 2] = {tileDesc_h->lo().I()+1,tileDesc_h->hi().I()+1, tileDesc_h->lo().J()+1,tileDesc_h->hi().J()+1, tileDesc_h->lo().K()+1,tileDesc_h->hi().K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_interior_p)) + n * SIZE_TILE_INTERIOR;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_interior_h), SIZE_TILE_INTERIOR);

        int _tile_lbound_h[MILHOJA_MDIM] = { lbound.I()+1, lbound.J()+1, lbound.K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_lbound_p)) + n * SIZE_TILE_LBOUND;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_lbound_h), SIZE_TILE_LBOUND);

        int _tile_ubound_h[MILHOJA_MDIM] = { ubound.I()+1, ubound.J()+1, ubound.K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_ubound_p)) + n * SIZE_TILE_UBOUND;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_ubound_h), SIZE_TILE_UBOUND);

        int _tile_arrayBounds_h[MILHOJA_MDIM * 2] = {tileDesc_h->loGC().I()+1,tileDesc_h->hiGC().I()+1, tileDesc_h->loGC().J()+1,tileDesc_h->hiGC().J()+1, tileDesc_h->loGC().K()+1,tileDesc_h->hiGC().K()+1 };
        char_ptr = static_cast<char*>(static_cast<void*>(_tile_arrayBounds_p)) + n * SIZE_TILE_ARRAYBOUNDS;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_tile_arrayBounds_h), SIZE_TILE_ARRAYBOUNDS);

        int _lbdd_CC_1_h[4] = {(lbound.I()) + 1,(lbound.J()) + 1,(lbound.K()) + 1,1};
        char_ptr = static_cast<char*>(static_cast<void*>(_lbdd_CC_1_p)) + n * SIZE_LBDD_CC_1;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_lbdd_CC_1_h), SIZE_LBDD_CC_1);
        
        int _lbdd_scratch_hydro_op1_auxC_h[3] = {(lo.I()-1) + 1,(lo.J()- 1) + 1,(lo.K()- 1) + 1};
        char_ptr = static_cast<char*>(static_cast<void*>(_lbdd_scratch_hydro_op1_auxC_p)) + n * SIZE_LBDD_SCRATCH_HYDRO_OP1_AUXC;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(_lbdd_scratch_hydro_op1_auxC_h), SIZE_LBDD_SCRATCH_HYDRO_OP1_AUXC);
        

        
        real* CC_1_d = tileDesc_h->dataPtr();
        constexpr std::size_t offset_CC_1 = (16 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (16 + 2 * 1 * MILHOJA_K3D) * static_cast<std::size_t>(0);
        constexpr std::size_t nBytes_CC_1 = (16 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (16 + 2 * 1 * MILHOJA_K3D) * ( 8 - 0 + 1 ) * sizeof(real);
        char_ptr = static_cast<char*>( static_cast<void*>(_CC_1_p) ) + n * SIZE_CC_1;
        std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(CC_1_d + offset_CC_1), nBytes_CC_1);
        
        
        
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[DataPacket_gpu_tf_hydro pack] Unable to acquire stream 1.");
    stream2_ = RuntimeBackend::instance().requestStream(true);
    if(!stream2_.isValid())
    	throw std::runtime_error("[DataPacket_gpu_tf_hydro::pack] Unable to acquire stream 2.");
    stream3_ = RuntimeBackend::instance().requestStream(true);
    if(!stream3_.isValid())
    	throw std::runtime_error("[DataPacket_gpu_tf_hydro::pack] Unable to acquire stream 3.");
    
}

void DataPacket_gpu_tf_hydro::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0)
        throw std::logic_error("[DataPacket_gpu_tf_hydro unpack] Empty data packet.");
    if (!stream_.isValid())
        throw std::logic_error("[DataPacket_gpu_tf_hydro unpack] Stream not acquired.");
    RuntimeBackend::instance().releaseStream(stream_);
    assert(!stream_.isValid());
    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        real* CC_1_data_h = tileDesc_h->dataPtr();
        
        real* CC_1_data_p = static_cast<real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( _CC_1_p ) ) + n * SIZE_CC_1 ) );
        
        constexpr std::size_t offset_CC_1 = (16 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (16 + 2 * 1 * MILHOJA_K3D) * static_cast<std::size_t>(0);
        real*        start_h_CC_1 = CC_1_data_h + offset_CC_1;
        const real*  start_p_CC_1 = CC_1_data_p + offset_CC_1;
        constexpr std::size_t nBytes_CC_1 = (16 + 2 * 1 * MILHOJA_K1D) * (16 + 2 * 1 * MILHOJA_K2D) * (16 + 2 * 1 * MILHOJA_K3D) * ( 7 - 0 + 1 ) * sizeof(real);
        std::memcpy(static_cast<void*>(start_h_CC_1), static_cast<const void*>(start_p_CC_1), nBytes_CC_1);
        
        
        
    }
}
