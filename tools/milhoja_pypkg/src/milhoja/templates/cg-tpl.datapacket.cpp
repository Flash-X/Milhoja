/* _connector:datapacket */
#include "_param:header_name"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_RuntimeBackend.h>

#if 0
/* _link:public_members */
/* _link:tileconst_members */
/* _link:tileconst_args */
/* _link:size_determination */
/* _link:includes */
/* _link:stream_functions_h */
/* _link:memcpy_tile_scratch */
/* _link:extra_streams*/
/* _link:omp_requires */
#endif

std::unique_ptr<milhoja::DataPacket> _param:class_name::clone(void) const {
    return std::unique_ptr<milhoja::DataPacket>{
        new _param:class_name {
            /* _link:host_members */
        }
    };
}

// Constructor arguments for DataPacket classes are copied by value into non-reference data members. 
// Thus, these values are frozen at instantiation. 
_param:class_name::_param:class_name(
    /* _link:constructor_args */
)
    : 
    milhoja::DataPacket{},
    /* _link:set_members */
    /* _link:set_tileconst */
    /* _link:set_size_det */
    {
}

_param:class_name::~_param:class_name(void) {
    /* _link:destructor */
}

/* _link:stream_functions_cxx */

void _param:class_name::pack(void) {
    using namespace milhoja;
    std::string errMsg = isNull();
    if (errMsg != "")
        throw std::logic_error("[_param:class_name pack] " + errMsg);
    else if (tiles_.size() == 0)
        throw std::logic_error("[_param:class_name pack] No tiles added.");

    // note: cannot set ntiles in the constructor because tiles_ is not filled yet.
    /* _link:nTiles_value */

    constexpr std::size_t SIZE_CONSTRUCTOR = pad(
        /* _link:size_constructor */
    );
    if (SIZE_CONSTRUCTOR % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name pack] SIZE_CONSTRUCTOR padding failure");

    std::size_t SIZE_TILEMETADATA = pad( _nTiles_h * ( 
        /* _link:size_tile_metadata */
    ));
    if (SIZE_TILEMETADATA % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name pack] SIZE_TILEMETADATA padding failure");
      
    std::size_t SIZE_TILEIN = pad( _nTiles_h * (
        /* _link:size_tile_in */
    ));
    if (SIZE_TILEIN % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name pack] SIZE_TILEIN padding failure");

    std::size_t SIZE_TILEINOUT = pad( _nTiles_h * (
        /* _link:size_tile_in_out */
    ));
    if (SIZE_TILEINOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name pack] SIZE_TILEINOUT padding failure");

    std::size_t SIZE_TILEOUT = pad( _nTiles_h * (
        /* _link:size_tile_out */
    ));
    if (SIZE_TILEOUT % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name pack] SIZE_TILEOUT padding failure");

    std::size_t SIZE_TILESCRATCH = pad( _nTiles_h * (
        /* _link:size_tile_scratch */
    ));
    if (SIZE_TILESCRATCH % ALIGN_SIZE != 0)
        throw std::logic_error("[_param:class_name pack] SIZE_TILESCRATCH padding failure");

    nCopyToGpuBytes_ = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT;
    nReturnToHostBytes_ = SIZE_TILEINOUT + SIZE_TILEOUT;
    std::size_t nBytesPerPacket = SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN + SIZE_TILEINOUT + SIZE_TILEOUT;
    RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket, &packet_p_, nBytesPerPacket + SIZE_TILESCRATCH, &packet_d_);
	
    // pointer determination phase
    static_assert(sizeof(char) == 1);
    char* ptr_d = static_cast<char*>(packet_d_);
    
    /* _link:pointers_tile_scratch */
    copyInStart_p_ = static_cast<char*>(packet_p_);
    copyInStart_d_ = static_cast<char*>(packet_d_) + SIZE_TILESCRATCH;
    char* ptr_p = copyInStart_p_;
    ptr_d = copyInStart_d_;

    /* _link:pointers_constructor */
    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR;
    /* _link:pointers_tile_metadata */
    ptr_p = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    ptr_d = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA;
    /* _link:pointers_tile_in */
    copyInOutStart_p_ = copyInStart_p_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    copyInOutStart_d_ = copyInStart_d_ + SIZE_CONSTRUCTOR + SIZE_TILEMETADATA + SIZE_TILEIN;
    ptr_p = copyInOutStart_p_;
    ptr_d = copyInOutStart_d_;
    /* _link:pointers_tile_in_out */
    char* copyOutStart_p_ = copyInOutStart_p_ + SIZE_TILEINOUT;
    char* copyOutStart_d_ = copyInOutStart_d_ + SIZE_TILEINOUT;
    /* _link:pointers_tile_out */
    //memcopy phase
    /* _link:memcpy_constructor */
    char* char_ptr;
    for (auto n = 0; n < _nTiles_h; n++) {
        Tile* tileDesc_h = tiles_[n].get();
        if (tileDesc_h == nullptr) throw std::runtime_error("[_param:class_name pack] Bad tiledesc.");
        /* _link:tile_descriptor */
        /* _link:memcpy_tile_metadata */
        /* _link:memcpy_tile_in */
        /* _link:memcpy_tile_in_out */
        /* _link:memcpy_tile_out */
    }

    stream_ = RuntimeBackend::instance().requestStream(true);
    if (!stream_.isValid())
        throw std::runtime_error("[_param:class_name pack] Unable to acquire stream 1.");
    /* _link:n_extra_streams */
}

void _param:class_name::unpack(void) {
    using namespace milhoja;
    if (tiles_.size() <= 0) 
        throw std::logic_error("[_param:class_name unpack] Empty data packet.");
    if (!stream_.isValid()) 
        throw std::logic_error("[_param:class_name unpack] Stream not acquired.");
    RuntimeBackend::instance().releaseStream(stream_);
    assert(!stream_.isValid());
    for (auto n = 0; n < _nTiles_h; ++n) {
        Tile* tileDesc_h = tiles_[n].get();
        /* _link:in_pointers */
        /* _link:out_pointers */
        /* _link:unpack_tile_in_out */
        /* _link:unpack_tile_out */
    }
}

