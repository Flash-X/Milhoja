/* _connector:datapacket */

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
/* _link:public_members */
/* _link:includes */
/* _link:stream_functions_h */
/* _link:memcpy_tilescratch */
/* _link:extra_streams*/
#endif

std::unique_ptr<milhoja::DataPacket> _param:i_give_up::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new _param:i_give_up {
            /* _link:host_members */
        }
    };
}

_param:i_give_up::_param:i_give_up(
    /* _link:constructor_args */
)
    : 
    /* _link:set_members */ 
    {
    // std::cout << "Hello, World!" << std::endl;
}

_param:i_give_up::~_param:i_give_up(void) {
    // std::cout << "Goodbye, cruel World!" << std::endl;
    /* _link:destructor */
    nullify();
}

/* _link:stream_functions_cxx */

void _param:i_give_up::pack(void) {
    using namespace milhoja;
	std::string errMsg = isNull();
	if (errMsg != "")
		throw std::logic_error("[_param:class_name::pack] " + errMsg);
	else if (tiles_.size() == 0)
		throw std::logic_error("[_param:class_name::pack] No tiles added.");
    static_assert(sizeof(char) == 1);


    _nTiles_h = tiles_.size();
    // size determination
    /* _link:size_determination */

	std::size_t SIZE_CONSTRUCTOR = pad(
        /* _link:size_constructor */
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name] SIZE_CONSTRUCTOR padding failure");

//    std::cout << "Constructor size: " << SIZE_CONSTRUCTOR << std::endl;

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * ( 
        /* _link:size_tilemetadata */
        )
    );
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name] SIZE_TILEMETADATA padding failure");

//    std::cout << "Tilemetadata size: " << SIZE_TILEMETADATA << std::endl;
        
    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
        /* _link:size_tilein */
        )
    );
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name] SIZE_TILEIN padding failure");

//    std::cout << "Tilein size: " << SIZE_TILEIN << std::endl;

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
        /* _link:size_tileinout */
        )
    );
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name] SIZE_TILEINOUT padding failure");

//    std::cout << "Tileinout size: " << SIZE_TILEINOUT << std::endl;

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
        /* _link:size_tileout */
        )
    );
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name] SIZE_TILEOUT padding failure");

//    std::cout << "Tileout size: " << SIZE_TILEOUT << std::endl;

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
        /* _link:size_tilescratch */
        )
    );
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name] SIZE_TILESCRATCH padding failure");

//    std::cout << "Tilescratch size: " << SIZE_TILESCRATCH << std::endl;

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT + SIZE_TILESCRATCH;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - SIZE_TILESCRATCH, &packet_p_, nBytesPerPacket, &packet_d_);
	
    // pointer determination phase
    // std::size_t ptr_d = 0;
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);

    /* _link:pointers_tilescratch */

    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ = static_cast<char*>(packet_d_) + SIZE_TILESCRATCH;
    char* ptr_p = copyInStart_p_;
    ptr_d = copyInStart_d_;
    
    /* _link:pointers_constructor */

    /* _link:pointers_tilemetadata */

    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;

    /* _link:pointers_tilein */

    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;

    /* _link:pointers_tileinout */
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;

    /* _link:pointers_tileout */

    //memcopy phase
    /* _link:memcpy_constructor */

    char* char_ptr;
    for (std::size_t n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[pack] Bad tiledesc.");
        /* _link:tile_descriptor */

        /* _link:memcpy_tilemetadata */
        /* _link:memcpy_tilein */
        /* _link:memcpy_tileinout */
        /* _link:memcpy_tileout */
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[pack] Unable to acquire stream 1.");
    /* _link:nextrastreams */
}

void _param:i_give_up::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0) throw std::logic_error("[unpack] Empty data packet.");
    if (!stream_.isValid()) throw std::logic_error("[unpack] Stream not acquired.");
    if (pinnedPtrs_ == nullptr) throw std::logic_error("[unpack] No pinned pointers set.");
    RuntimeBackend::instance().releaseStream(stream_);
	assert(!stream_.isValid());

    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        /* _link:in_pointers */
        /* _link:out_pointers */

        /* _link:unpack_tileinout */
        /* _link:unpack_tileout */
    }
}

// int main() {
//     //this is acting as driver_evolveAll
//     _param:class_name packet{2.1};
//     //this is driver_evolveAll calling runtime
//     packet.pack();
//     packet.unpack();
//     return 0;
// }
