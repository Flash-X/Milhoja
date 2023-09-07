/* _connector:datapacket */
#ifndef _param:ndef_name
#define _param:ndef_name

#if 0
/* _link:set_members */
/* _link:stream_functions_cxx */
/* _link:destructor */
/* _link:size_determination */
/* _link:pointers_tile_scratch */
/* _link:pointers_constructor */
/* _link:pointers_tile_metadata */
/* _link:pointers_tile_in */
/* _link:pointers_tile_in_out */
/* _link:pointers_tile_out */
/* _link:memcpy_constructor */
/* _link:tile_descriptor */
/* _link:memcpy_tile_metadata */
/* _link:memcpy_tile_in */
/* _link:memcpy_tile_in_out */
/* _link:memcpy_tile_out */
/* _link:memcpy_tile_scratch */
/* _link:section_sizes */
/* _link:n_extra_streams */
/* _link:unpack_tile_in_out */
/* _link:unpack_tile_out */
/* _link:size_constructor */
/* _link:size_tile_metadata */
/* _link:size_tile_in */
/* _link:size_tile_in_out */
/* _link:size_tile_out */
/* _link:host_members */
/* _link:size_tile_scratch */
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

using real = milhoja::Real;
using milhoja::FArray4D;
using milhoja::Stream;
using milhoja::IntVect;
using milhoja::RealVect;

class _param:class_name : public milhoja::DataPacket {
public:
    // constructor
    _param:class_name(
        /* _link:constructor_args */
    );
    // destructor
    ~_param:class_name(void);

    //helper methods from base DataPacket class.
    std::unique_ptr<milhoja::DataPacket> clone(void) const override;
    _param:class_name(_param:class_name&) = delete;
    _param:class_name(const _param:class_name&) = delete;
    _param:class_name(_param:class_name&& packet) = delete;
    _param:class_name& operator=(_param:class_name&)       = delete;
	_param:class_name& operator=(const _param:class_name&) = delete;
	_param:class_name& operator=(_param:class_name&& rhs)  = delete;
    
    // pack and unpack functions from base class. 
    void pack(void) override;
    void unpack(void) override;

    // DataPacket members are made public so a matching task function can easily access them.
    // Since both files are auto-generated and not maintained by humans, this is fine.
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

