
#include <iostream>
#include "DataPacket_Hydro_gpu_3.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_Grid.h>
#include <Milhoja_RuntimeBackend.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>

#if 0
Real _dt_h;
Real* _dt_d;
int _nTiles_h;
int* _nTiles_d;
RealVect* _deltas_d;
IntVect* _lo_d;
IntVect* _hi_d;
IntVect* _loAuxC_d;
IntVect* _loU_d;
IntVect* _loFl_d;
Real* _U_d;
Real* U_p;
Real* _auxC_d;
Real* _flX_d;
Real* _flY_d;
Real* _flZ_d;
FArray4D* _f4_U_d;
FArray4D* _f4_auxC_d;
FArray4D* _f4_flX_d;
FArray4D* _f4_flY_d;
FArray4D* _f4_flZ_d;


#endif

std::unique_ptr<milhoja::DataPacket> cgkit_dataPacket_Hydro_gpu_3_2nd_pass::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new cgkit_dataPacket_Hydro_gpu_3_2nd_pass {
      _dt_h
      
        }
    };
}

cgkit_dataPacket_Hydro_gpu_3_2nd_pass::cgkit_dataPacket_Hydro_gpu_3_2nd_pass(
Real dt

)
    :
_dt_h{dt},
_dt_d{nullptr},
_nTiles_h{tiles_.size()},
_nTiles_d{nullptr},
_deltas_d{nullptr},
_lo_d{nullptr},
_hi_d{nullptr},
_loAuxC_d{nullptr},
_loU_d{nullptr},
_loFl_d{nullptr},
_U_d{nullptr},
_auxC_d{nullptr},
_flX_d{nullptr},
_flY_d{nullptr},
_flZ_d{nullptr}

    {
    // std::cout << "Hello, World!" << std::endl;
}

cgkit_dataPacket_Hydro_gpu_3_2nd_pass::~cgkit_dataPacket_Hydro_gpu_3_2nd_pass(void) {
    // std::cout << "Goodbye, cruel World!" << std::endl;
    nullify();
}


void cgkit_dataPacket_Hydro_gpu_3_2nd_pass::pack(void) {
    using namespace milhoja;
	std::string errMsg = isNull();
	if (errMsg != "")
		throw std::logic_error("[packet::pack] " + errMsg);
	else if (tiles_.size() == 0)
		throw std::logic_error("[packet::pack] No tiles added.");
    static_assert(sizeof(char) == 1);


    _nTiles_h = tiles_.size();
    // size determination
  constexpr std::size_t SIZE_DT = pad( sizeof(Real) );
  constexpr std::size_t SIZE_NTILES = pad( sizeof(int) );
  constexpr std::size_t SIZE_TILE_DELTAS = sizeof(RealVect);
  constexpr std::size_t SIZE_TILE_LO = sizeof(IntVect);
  constexpr std::size_t SIZE_TILE_HI = sizeof(IntVect);
  constexpr std::size_t SIZE_LOAUXC = sizeof(IntVect);
  constexpr std::size_t SIZE_LOU = sizeof(IntVect);
  constexpr std::size_t SIZE_LOFL = sizeof(IntVect);
  constexpr std::size_t SIZE_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * (8 - 0 + 1) * sizeof(Real);
  constexpr std::size_t SIZE_AUXC = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * sizeof(Real);
  constexpr std::size_t SIZE_FLX = ((8 + 2 * 1) + 1) * (8 + 2 * 1) * (1 + 2 * 0) * (5) * sizeof(Real);
  constexpr std::size_t SIZE_FLY = (8 + 2 * 1) * ((8 + 2 * 1) + 1) * (1 + 2 * 0) * (5) * sizeof(Real);
  constexpr std::size_t SIZE_FLZ = (1) * (1) * (1) * (1) * sizeof(Real);
  

	std::size_t SIZE_CONSTRUCTOR = pad(
  SIZE_DT + SIZE_NTILES + _nTiles_h * sizeof(PacketContents)
  
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[packet] SIZE_CONSTRUCTOR padding failure");

//    std::cout << "Constructor size: " << SIZE_CONSTRUCTOR << std::endl;

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * (
  (5 * sizeof(FArray4D)) + SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI + SIZE_LOAUXC + SIZE_LOU + SIZE_LOFL
  
        )
    );
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[packet] SIZE_TILEMETADATA padding failure");

//    std::cout << "Tilemetadata size: " << SIZE_TILEMETADATA << std::endl;

    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
  0
  
        )
    );
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[packet] SIZE_TILEIN padding failure");

//    std::cout << "Tilein size: " << SIZE_TILEIN << std::endl;

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
  SIZE_U
  
        )
    );
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[packet] SIZE_TILEINOUT padding failure");

//    std::cout << "Tileinout size: " << SIZE_TILEINOUT << std::endl;

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
  0
  
        )
    );
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[packet] SIZE_TILEOUT padding failure");

//    std::cout << "Tileout size: " << SIZE_TILEOUT << std::endl;

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
  SIZE_AUXC + SIZE_FLX + SIZE_FLY + SIZE_FLZ
  
        )
    );
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[packet] SIZE_TILESCRATCH padding failure");

//    std::cout << "Tilescratch size: " << SIZE_TILESCRATCH << std::endl;

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT + SIZE_TILESCRATCH;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - SIZE_TILESCRATCH, &packet_p_, nBytesPerPacket, &packet_d_);

    // pointer determination phase
    // std::size_t ptr_d = 0;
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);

  _auxC_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_d+= _nTiles_h * SIZE_AUXC;
  
  _flX_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_d+= _nTiles_h * SIZE_FLX;
  
  _flY_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_d+= _nTiles_h * SIZE_FLY;
  
  _flZ_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_d+= _nTiles_h * SIZE_FLZ;
  
  

    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ = static_cast<char*>(packet_d_) + SIZE_TILESCRATCH;
    char* ptr_p = copyInStart_p_;
    ptr_d = copyInStart_d_;

  Real* dt_p = static_cast<Real*>( static_cast<void*>(ptr_p) );
  _dt_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_p+=SIZE_DT;
  ptr_d+=SIZE_DT;
  
  int* nTiles_p = static_cast<int*>( static_cast<void*>(ptr_p) );
  _nTiles_d = static_cast<int*>( static_cast<void*>(ptr_d) );
  ptr_p+=SIZE_NTILES;
  ptr_d+=SIZE_NTILES;
  
  

  RealVect* deltas_p = static_cast<RealVect*>( static_cast<void*>(ptr_p) );
  _deltas_d = static_cast<RealVect*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_TILE_DELTAS;
  ptr_d+=_nTiles_h * SIZE_TILE_DELTAS;
  
  IntVect* lo_p = static_cast<IntVect*>( static_cast<void*>(ptr_p) );
  _lo_d = static_cast<IntVect*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_TILE_LO;
  ptr_d+=_nTiles_h * SIZE_TILE_LO;
  
  IntVect* hi_p = static_cast<IntVect*>( static_cast<void*>(ptr_p) );
  _hi_d = static_cast<IntVect*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_TILE_HI;
  ptr_d+=_nTiles_h * SIZE_TILE_HI;
  
  IntVect* loAuxC_p = static_cast<IntVect*>( static_cast<void*>(ptr_p) );
  _loAuxC_d = static_cast<IntVect*>( static_cast<void*>(ptr_d) );
  ptr_p += _nTiles_h * SIZE_LOAUXC;
  ptr_d += _nTiles_h * SIZE_LOAUXC;
  
  IntVect* loU_p = static_cast<IntVect*>( static_cast<void*>(ptr_p) );
  _loU_d = static_cast<IntVect*>( static_cast<void*>(ptr_d) );
  ptr_p += _nTiles_h * SIZE_LOU;
  ptr_d += _nTiles_h * SIZE_LOU;
  
  IntVect* loFl_p = static_cast<IntVect*>( static_cast<void*>(ptr_p) );
  _loFl_d = static_cast<IntVect*>( static_cast<void*>(ptr_d) );
  ptr_p += _nTiles_h * SIZE_LOFL;
  ptr_d += _nTiles_h * SIZE_LOFL;
  
  char* _f4_U_p = ptr_p;
  _f4_U_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* _f4_auxC_p = ptr_p;
  _f4_auxC_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* _f4_flX_p = ptr_p;
  _f4_flX_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* _f4_flY_p = ptr_p;
  _f4_flY_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* _f4_flZ_p = ptr_p;
  _f4_flZ_d = static_cast<FArray4D*>( static_cast<void*>( ptr_d ) );
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  

    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;

  

    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;

   U_p = static_cast<Real*>( static_cast<void*>(ptr_p) );
  _U_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_U;
  ptr_d+=_nTiles_h * SIZE_U;
  
  
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;

  

    //memcopy phase
  std::memcpy(dt_p, static_cast<void*>(&_dt_h), SIZE_DT);
  std::memcpy(nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);
  

    char* char_ptr;
    for (std::size_t n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[pack] Bad tiledesc.");
    const RealVect deltas = tileDesc_h->deltas();
    const IntVect lo = tileDesc_h->lo();
    const IntVect hi = tileDesc_h->hi();
    const IntVect hiGC = tileDesc_h->hiGC();
    const IntVect loGC = tileDesc_h->loGC();
    const IntVect loAuxC = tileDesc_h->lo() - IntVect{ LIST_NDIM(1, 1, 0) };
    const IntVect loU = tileDesc_h->loGC();
    const IntVect loFl = tileDesc_h->lo();
    

    char_ptr = static_cast<char*>( static_cast<void*>( deltas_p ) ) + n * SIZE_TILE_DELTAS;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&deltas), SIZE_TILE_DELTAS);
    
    char_ptr = static_cast<char*>( static_cast<void*>( lo_p ) ) + n * SIZE_TILE_LO;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&lo), SIZE_TILE_LO);
    
    char_ptr = static_cast<char*>( static_cast<void*>( hi_p ) ) + n * SIZE_TILE_HI;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&hi), SIZE_TILE_HI);
    
    char_ptr = static_cast<char*>( static_cast<void*>( loAuxC_p ) ) + n * SIZE_LOAUXC;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&loAuxC), SIZE_LOAUXC);
    
    char_ptr = static_cast<char*>( static_cast<void*>( loU_p ) ) + n * SIZE_LOU;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&loU), SIZE_LOU);
    
    char_ptr = static_cast<char*>( static_cast<void*>( loFl_p ) ) + n * SIZE_LOFL;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&loFl), SIZE_LOFL);
    
    
    char_ptr = static_cast<char*>( static_cast<void*>(_f4_U_d) ) + n * sizeof(FArray4D);
    FArray4D U_device{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_U_d) ) + n * SIZE_U)), loGC, hiGC, 8 - 0 + 1};
    char_ptr = _f4_U_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&U_device), sizeof(FArray4D));
    
    char_ptr = static_cast<char*>( static_cast<void*>(_f4_auxC_d) ) + n * sizeof(FArray4D);
    FArray4D auxC_device{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_auxC_d) ) + n * SIZE_AUXC)), loGC, hiGC, 1};
    char_ptr = _f4_auxC_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&auxC_device), sizeof(FArray4D));
    
    char_ptr = static_cast<char*>( static_cast<void*>(_f4_flX_d) ) + n * sizeof(FArray4D);
    FArray4D flX_device{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flX_d) ) + n * SIZE_FLX)), lo, IntVect{ LIST_NDIM( hi.I()+1, hi.J(), hi.K() ) }, 5};
    char_ptr = _f4_flX_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flX_device), sizeof(FArray4D));
    
    char_ptr = static_cast<char*>( static_cast<void*>(_f4_flY_d) ) + n * sizeof(FArray4D);
    FArray4D flY_device{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flY_d) ) + n * SIZE_FLY)), lo, IntVect{ LIST_NDIM( hi.I(), hi.J()+1, hi.K() ) }, 5};
    char_ptr = _f4_flY_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flY_device), sizeof(FArray4D));
    
    char_ptr = static_cast<char*>( static_cast<void*>(_f4_flZ_d) ) + n * sizeof(FArray4D);
    FArray4D flZ_device{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flZ_d) ) + n * SIZE_FLZ)), lo, IntVect{ LIST_NDIM( hi.I(), hi.J(), hi.K()+1 ) }, 1};
    char_ptr = _f4_flZ_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flZ_device), sizeof(FArray4D));
    
    
    Real* U_d = tileDesc_h->dataPtr();
    std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
    std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 8 - 0 + 1 ) * sizeof(Real);
    char_ptr = static_cast<char*>( static_cast<void*>(U_p) ) + n * SIZE_U;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(U_d + offset_U), nBytes_U);
    
    
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[pack] Unable to acquire stream 1.");
}

void cgkit_dataPacket_Hydro_gpu_3_2nd_pass::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0) throw std::logic_error("[unpack] Empty data packet.");
    if (!stream_.isValid()) throw std::logic_error("[unpack] Stream not acquired.");
    // if (pinnedPtrs_ == nullptr) throw std::logic_error("[unpack] No pinned pointers set.");
    RuntimeBackend::instance().releaseStream(stream_);
	assert(!stream_.isValid());

  constexpr std::size_t SIZE_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * (8 - 0 + 1) * sizeof(Real);
  

    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
    Real* U_data_h = tileDesc_h->dataPtr();
    
    Real* U_data_p = static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>( U_p ) ) + n * SIZE_U ) );
    

    Real* U_d = tileDesc_h->dataPtr();
    std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
    Real*        start_h_U = U_data_h + offset_U;
    const Real*  start_p_U = U_data_p + offset_U;
    std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 7 - 0 + 1 ) * sizeof(RealVect);
    std::memcpy(static_cast<void*>(start_h_U), static_cast<const void*>(start_p_U), nBytes_U);
    
    
    }
}

// int main() {
//     //this is acting as driver_evolveAll
//     //this is driver_evolveAll calling runtime
//     packet.pack();
//     packet.unpack();
//     return 0;
// }