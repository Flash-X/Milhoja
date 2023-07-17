/* _connector:datapacket */
#ifndef _param:ndef_name
#define _param:ndef_name

#if 0
/* _link:set_members */
/* _link:stream_functions_cxx */
/* _link:destructor */
/* _link:size_determination */
/* _link:pointers_tilescratch */
/* _link:pointers_constructor */
/* _link:pointers_tilemetadata */
/* _link:pointers_tilein */
/* _link:pointers_tileinout */
/* _link:pointers_tileout */
/* _link:memcpy_constructor */
/* _link:tile_descriptor */
/* _link:memcpy_tilemetadata */
/* _link:memcpy_tilein */
/* _link:memcpy_tileinout */
/* _link:memcpy_tileout */
/* _link:memcpy_tilescratch */
/* _link:section_sizes */
/* _link:nextrastreams */
/* _link:unpack_tileinout */
/* _link:unpack_tileout */
/* _link:size_constructor */
/* _link:size_tilemetadata */
/* _link:size_tilein */
/* _link:size_tileinout */
/* _link:size_tileout */
/* _link:host_members */
/* _link:size_tilescratch */
/* _link:in_pointers */
/* _link:out_pointers */
/* _link:pinned_sizes */
#endif

/* _link:includes */
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>

using milhoja::Real;
using milhoja::FArray4D;
using milhoja::Stream;
using milhoja::IntVect;
using milhoja::RealVect;

class _param:class_name : public milhoja::DataPacket {
// class _param:class_name { 
public:
    //constructor / destructor
    _param:class_name(
        /* _link:constructor_args */
    );
    ~_param:class_name(void);

    //helper methods
    std::unique_ptr<milhoja::DataPacket> clone(void) const;
    _param:class_name(_param:class_name&) = delete;
    _param:class_name(const _param:class_name&) = delete;
    _param:class_name(_param:class_name&& packet) = delete;
    _param:class_name& operator=(_param:class_name&)       = delete;
	_param:class_name& operator=(const _param:class_name&) = delete;
	_param:class_name& operator=(_param:class_name&& rhs)  = delete;
    
    void pack(void);
    void unpack(void);

    /* _link:stream_functions_h */
    /* _link:public_members */
private:
    static constexpr std::size_t ALIGN_SIZE=_param:align_size;
    static constexpr std::size_t pad(const std::size_t size) {
        return (((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    }

    /* _link:extra_streams */
};

#endif

