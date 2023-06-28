
#include <iostream>
#include "cgkit.datapacket.h"
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
Real* _deltas_d;
int* _lo_d;
int* _hi_d;
Real* _U_d;
Real* _auxC_d;
Real* _flX_d;
Real* _flY_d;
Real* _flZ_d;


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
  (5 * sizeof(FArray4D)) + SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI
  
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
  
  

    location_ = PacketDataLocation::CC1;
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
  
  

  Real* deltas_p = static_cast<Real*>( static_cast<void*>(ptr_p) );
  _deltas_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_TILE_DELTAS;
  ptr_d+=_nTiles_h * SIZE_TILE_DELTAS;
  
  int* lo_p = static_cast<int*>( static_cast<void*>(ptr_p) );
  _lo_d = static_cast<int*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_TILE_LO;
  ptr_d+=_nTiles_h * SIZE_TILE_LO;
  
  int* hi_p = static_cast<int*>( static_cast<void*>(ptr_p) );
  _hi_d = static_cast<int*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_TILE_HI;
  ptr_d+=_nTiles_h * SIZE_TILE_HI;
  
  char* U_fa4_p = ptr_p;
  char* U_fa4_d = ptr_d;
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* auxC_fa4_p = ptr_p;
  char* auxC_fa4_d = ptr_d;
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* flX_fa4_p = ptr_p;
  char* flX_fa4_d = ptr_d;
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* flY_fa4_p = ptr_p;
  char* flY_fa4_d = ptr_d;
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  char* flZ_fa4_p = ptr_p;
  char* flZ_fa4_d = ptr_d;
  ptr_p += _nTiles_h * sizeof(FArray4D);
  ptr_d += _nTiles_h * sizeof(FArray4D);
  
  

    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;

  

    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;

  Real* U_p = static_cast<Real*>( static_cast<void*>(ptr_p) );
  _U_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
  ptr_p+=_nTiles_h * SIZE_U;
  ptr_d+=_nTiles_h * SIZE_U;
  
  
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;

  

    //memcopy phase
  std::memcpy(dt_p, static_cast<void*>(&_dt_h), SIZE_DT);
  std::memcpy(nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);
  

    if (pinnedPtrs_) throw std::logic_error("[pack] Pinned pointers already exist");
	pinnedPtrs_ = new BlockPointersPinned[_nTiles_h];
	PacketContents* tilePtrs_p = contents_p_;

    char* char_ptr;
    for (std::size_t n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[pack] Bad tiledesc.");
    const RealVect deltas = tileDesc_h->deltas();
    const IntVect lo = tileDesc_h->lo();
    const IntVect hi = tileDesc_h->hi();
    const IntVect hiGC = tileDesc_h->hiGC();
    const IntVect loGC = tileDesc_h->loGC();
    
        Real* data_h = tileDesc_h->dataPtr();
        if (data_h == nullptr) throw std::logic_error("[pack] Invalid ptr to data in host.");

    char_ptr = static_cast<char*>( static_cast<void*>( deltas_p ) ) + n * SIZE_TILE_DELTAS;
    tilePtrs_p->deltas_d = static_cast<RealVect*>( static_cast<void*>(char_ptr) );
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&deltas), SIZE_TILE_DELTAS);
    
    char_ptr = static_cast<char*>( static_cast<void*>( lo_p ) ) + n * SIZE_TILE_LO;
    tilePtrs_p->lo_d = static_cast<IntVect*>( static_cast<void*>(char_ptr) );
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&lo), SIZE_TILE_LO);
    
    char_ptr = static_cast<char*>( static_cast<void*>( hi_p ) ) + n * SIZE_TILE_HI;
    tilePtrs_p->hi_d = static_cast<IntVect*>( static_cast<void*>(char_ptr) );
    std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(&hi), SIZE_TILE_HI);
    
    
    char_ptr = U_fa4_d + n * sizeof(FArray4D);
    tilePtrs_p->U_d = static_cast<FArray4D*>( static_cast<void*>(char_ptr) );
    FArray4D U_d{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_U_d) ) + n * SIZE_U)) };
    char_ptr = U_fa4_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&U_d), sizeof(FArray4D));
    
    char_ptr = auxC_fa4_d + n * sizeof(FArray4D);
    tilePtrs_p->auxC_d = static_cast<FArray4D*>( static_cast<void*>(char_ptr) );
    FArray4D auxC_d{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_auxC_d) ) + n * SIZE_AUXC)) };
    char_ptr = auxC_fa4_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&auxC_d), sizeof(FArray4D));
    
    char_ptr = flX_fa4_d + n * sizeof(FArray4D);
    tilePtrs_p->flX_d = static_cast<FArray4D*>( static_cast<void*>(char_ptr) );
    FArray4D flX_d{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flX_d) ) + n * SIZE_FLX)) };
    char_ptr = flX_fa4_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flX_d), sizeof(FArray4D));
    
    char_ptr = flY_fa4_d + n * sizeof(FArray4D);
    tilePtrs_p->flY_d = static_cast<FArray4D*>( static_cast<void*>(char_ptr) );
    FArray4D flY_d{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flY_d) ) + n * SIZE_FLY)) };
    char_ptr = flY_fa4_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flY_d), sizeof(FArray4D));
    
    char_ptr = flZ_fa4_d + n * sizeof(FArray4D);
    tilePtrs_p->flZ_d = static_cast<FArray4D*>( static_cast<void*>(char_ptr) );
    FArray4D flZ_d{ static_cast<Real*>( static_cast<void*>( static_cast<char*>( static_cast<void*>(_flZ_d) ) + n * SIZE_FLZ)) };
    char_ptr = flZ_fa4_p + n * sizeof(FArray4D);
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(&flZ_d), sizeof(FArray4D));
    
    
    std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
    std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 8 - 0 + 1 ) * sizeof(Real);
    char_ptr = static_cast<char*>( static_cast<void*>(U_p) ) + n * SIZE_U;
    std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(data_h + offset_U), nBytes_U);
    pinnedPtrs_[n].CC1_data = static_cast<Real*>( static_cast<void*>(char_ptr) );
    
    
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[pack] Unable to acquire stream 1.");
}

void cgkit_dataPacket_Hydro_gpu_3_2nd_pass::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0) throw std::logic_error("[unpack] Empty data packet.");
    if (!stream_.isValid()) throw std::logic_error("[unpack] Stream not acquired.");
    if (pinnedPtrs_ == nullptr) throw std::logic_error("[unpack] No pinned pointers set.");
    RuntimeBackend::instance().releaseStream(stream_);
	assert(!stream_.isValid());

    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        Real* data_h = tileDesc_h->dataPtr();
        const Real* data_p = pinnedPtrs_[n].CC1_data;
        if(data_h == nullptr) throw std::logic_error("[unpack] Invalid pointer to data in host.");
        if(data_p == nullptr) throw std::logic_error("[unpack] Invalid pointer to data in pinned.");

    std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
    Real*        start_h_U = data_h + offset_U;
    const Real*  start_p_U = data_p + offset_U;
    std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 7 - 0 + 1 ) * sizeof(Real);
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