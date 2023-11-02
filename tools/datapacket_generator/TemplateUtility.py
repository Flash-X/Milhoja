"""
This is the main script for generating a new DataPacket. The overall structure of this 
code is to move through every possible DataPacket JSON section and fill out the various CGKkit dictionaries
for generating every template.

TODO: How to sort bound class members by size?
TODO: Eventually logging should be more informative and replace all print statements.
"""
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from abc import abstractstaticmethod
from abc import abstractmethod
from DataPacketMemberVars import DataPacketMemberVars
from milhoja import TaskFunction
from milhoja import AbcCodeGenerator

class TemplateUtility():
    _EXT = "constructor"
    _T_SCRATCH = "tile_scratch"
    _T_MDATA = "tile_metadata"
    _T_IN = "tile_in"
    _T_IN_OUT = "tile_in_out"
    _T_OUT = "tile_out"

    _NTILES_VALUE = 'nTiles_value'
    _CON_ARGS = 'constructor_args'
    _SET_MEMBERS = 'set_members'
    _SIZE_DET = 'size_determination'
    _HOST_MEMBERS = 'host_members'
    _PUB_MEMBERS = 'public_members'
    _IN_PTRS = 'in_pointers'
    _OUT_PTRS = 'out_pointers'
    _T_DESCRIPTOR = 'tile_descriptor'
    _EXTENTS = "extents"
    _LBOUND = "lbound"

    _STREAM_FUNCS_H = 'stream_functions_h'
    _EXTRA_STREAMS = 'extra_streams'
    _EXTRA_STREAMS_PACK = "n_extra_streams"
    _DESTRUCTOR = 'destructor'
    _STREAM_FUNCS_CXX = 'stream_functions_cxx'

    _SOURCE_TILE_DATA_MAPPING = {
        "CENTER": "tileDesc_h->dataPtr()",
        "FLUXX": "&tileDesc_h->fluxData(milhoja::Axis::I)",
        "FLUXY": "&tileDesc_h->fluxData(milhoja::Axis::J)",
        "FLUXZ": "&tileDesc_h->fluxData(milhoja::Axis::K)"
    }

    @classmethod
    @abstractmethod
    def iterate_externals(cls, connectors: dict, size_connectors: dict, externals: OrderedDict):
        ...

    @classmethod
    @abstractmethod
    def iterate_tile_metadata(cls, connectors: dict, size_connectors: dict, tilemetadata: OrderedDict, num_arrays: int):
        ...

    @classmethod
    @abstractmethod
    def tile_metadata_memcpy(cls):
        ...

    @classmethod
    @abstractmethod
    def iterate_tile_in(cls):
        ...
    
    @classmethod
    @abstractmethod
    def iterate_tile_in_out(cls):
        ...

    @classmethod
    @abstractmethod
    def iterate_tile_out(cls):
        ...

    @classmethod
    @abstractmethod
    def iterate_tile_scratch(cls):
        ...

    @staticmethod
    def add_size_parameter(name: str, section_dict: dict, connectors: dict):
        """
        Adds a size connector to the params dict that is passed in. The size
        connector dictionary stores every size of every item in the data packet
        for use with a CGKit template.

        :param str name: The name of the size connector. Usually the name of a section.
        :param dict section_dict: The section used to generate sizes. Contains all of the items in the section.
        :param dict connectors: The dictionary containing all connectors generated for the packet.
        :rtype: None
        """
        size = '0'
        if section_dict:
            size = ' + '.join(f'SIZE_{item.upper()}' for item in section_dict)
        connectors[f'size_{name}'] = size

    @staticmethod
    def section_creation(
        name: str, 
        section: OrderedDict, 
        connectors: dict, 
        size_connectors
    ):
        """
        Creates a section and sets the default value to an empty list. 
        It's assumed that the function that calls this method 
        will populate the dictionary using the same name.
        
        :param str name: The name of the section to create.
        :param dict section: The dictionary to get all data packet items from. 
        :param dict connectors: The dictionary containing all link connectors. 
        :param dict size_connectors: The dictionary containing all connectors that determine sizes for each variable in the data packet.
        """
        TemplateUtility.add_size_parameter(name, section, size_connectors)
        connectors[f'pointers_{name}'] = []
        connectors[f'memcpy_{name}'] = []

    @staticmethod
    def set_pointer_determination(
        connectors: dict,
        section: str,
        info: DataPacketMemberVars,
        item_is_member_variable=False
    ):
        """
        Stores the code for determining pointer offsets in the 
        DataPacket into the connectors dictionary to be used with CGKit.
        
        :param dict connectors: The dict containing all connectors needed for use with CGKit.
        :param str section: The section to use to determine the key for the pointer determination.
        :param DataPacketMemberVars info: Contains information for formatting the name to get variable names.
        :param bool item_is_member_variable: Flag if *item* pinned memory pointer is a member variable
        """
        dtype = info.dtype
        # If the item is a data packet member variable, we don't need
        # to specify a type name here.
        # Note that pointers to memory in the remote device are always
        # member variables, so it will never need a type name.
        if item_is_member_variable: dtype = ""
        else: dtype += "* "
        
        # insert items into boiler plate for the pointer determination phase for *section*.
        connectors[f'pointers_{section}'].append(
            f"{dtype}{info.pinned} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_p) );\n" + 
            f"{info.device} = static_cast<{info.dtype}*>( static_cast<void*>(ptr_d) );\n" + 
            f"ptr_p+={info.total_size};\n" + 
            f"ptr_d+={info.total_size};\n\n"
        )

    @classmethod
    def generate_extra_streams_information(cls, connectors: dict, extra_streams: int):
        """
        Fills the links extra_streams, stream_functions_h/cxx.
        TODO: Streams in the datapacket should use an array implementation

        :param dict connectors: The dictionary containing all connectors to be used with CGKit.
        :param int extra_streams: The number of extra streams specified in the data packet JSON.
        :rtype: None
        """
        if extra_streams < 1: return

        connectors[cls._STREAM_FUNCS_H].extend([
            f'int extraAsynchronousQueue(const unsigned int id) override;\n',
            f'void releaseExtraQueue(const unsigned int id) override;\n'
        ])
        connectors[cls._EXTRA_STREAMS].extend([
            f'milhoja::Stream stream{i}_;\n' for i in range(2, extra_streams+2)
        ])
        connectors[cls._DESTRUCTOR].extend([
            f'if (stream{i}_.isValid())\n' +
            '\tthrow std::logic_error("[_param:class_name::~_param:class_name] ' +
            f'Stream {i} not released");\n'
            for i in range(2, extra_streams+2)
        ])

        # these insert the various stream functions if there are more than 1 stream
        # normally these would be in the template but these functions have no reason to exist if the
        # task function does not use more than 1 stream.
        connectors[cls._STREAM_FUNCS_CXX].extend([
                f'int _param:class_name::extraAsynchronousQueue(const unsigned int id) {{\n',
                f'\tif (id > INT_MAX)\n\t\tthrow std::overflow_error("[_param:class_name extraAsynchronousQueue] id is too large for int.");\n'
                f'\tif((id < 2) || (id > {extra_streams} + 1))\n' +
                f'\t\tthrow std::invalid_argument("[_param:class_name::extraAsynchronousQueue] Invalid id.");\n\t',
            ] + [
                f'else if (id == {i}) {{\n' +
                f'\t\tif (!stream{i}_.isValid()) ' + 
                f'\n\t\t\tthrow std::logic_error("[_param:class_name::extraAsynchronousQueue] ' +
                f'Stream {i} invalid.");\n\t\treturn stream{i}_.accAsyncQueue;\n' + 
                f'\t}} '
                for i in range(2, extra_streams+2)
            ] + [
                '\n\treturn 0;\n',
                '}\n\n'
            ] + [
                f'void _param:class_name::releaseExtraQueue(const unsigned int id) {{\n',
                f'\tif((id < 2) || (id > {extra_streams} + 1))\n' + 
                f'\t\tthrow std::invalid_argument("[_param:class_name::releaseExtraQueue] Invalid id.");\n\t',
            ] + [
                f'else if(id == {i}) {{\n'
                f'\t\tif(!stream{i}_.isValid())\n' + 
                f'\t\t\tthrow std::logic_error("[_param:class_name::releaseExtraQueue] Stream {i} invalid.");\n' + 
                f'\t\tmilhoja::RuntimeBackend::instance().releaseStream(stream{i}_);\n' 
                f'\t}} '
                for i in range(2, extra_streams+2)
            ] + [   
                '\n'
                '}\n'
            ]
        )

        # Inserts the code necessary to acquire extra streams. 
        connectors[cls._EXTRA_STREAMS_PACK].extend([
            f'stream{i}_ = RuntimeBackend::instance().requestStream(true);\n' + 
            f'if(!stream{i}_.isValid())\n' +
            f'\tthrow std::runtime_error("[_param:class_name::pack] Unable to acquire stream {i}.");\n'
            for i in range(2, extra_streams+2)
        ])

    @classmethod
    def add_memcpy_connector(
        cls,
        connectors: dict, 
        section: str, 
        extents: str, item: str, 
        start: int, end: int, 
        size_item: str, raw_type: str,
        source: str
    ):
        """
        Adds a memcpy connector based on the information passed in.
        
        :param dict connectors: The dict containing all cgkit connectors
        :param str section: The section to add a memcpy connector for.
        :param str extents: The string containing array extents information.
        :param str item: The item to copy into pinned memory.
        :param int start: The starting index of the array.
        :param int end: The ending index of the array.
        :param str size_item: The string containing the size variable for *item*.
        :param str raw_type: The data type of the item.
        """
        offset = f"{extents} * static_cast<std::size_t>({start})"
        nBytes = f'{extents} * ( {end} - {start} + 1 ) * sizeof({raw_type})'
        
        # This exists inside the pack function for copying data from tile_in to the device.
        # Luckily we don't really need to use DataPacketMemberVars here because the temporary device pointer 
        # is locally scoped.

        data_pointer_string = cls._SOURCE_TILE_DATA_MAPPING[source.upper()]

        # TODO: Use grid data to get data pointer information.
        connectors[f'memcpy_{section}'].extend([
            f'{raw_type}* {item}_d = {data_pointer_string};\n'  # eventually we will pass arguments to data ptr for specific data args.
            f'constexpr std::size_t offset_{item} = {offset};\n',
            f'constexpr std::size_t nBytes_{item} = {nBytes};\n',
            f'char_ptr = static_cast<char*>( static_cast<void*>(_{item}_p) ) + n * {size_item};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>({item}_d + offset_{item}), nBytes_{item});\n\n'
        ])

    @classmethod
    def add_unpack_connector(
        cls,
        connectors: dict, 
        section: str, 
        extents, 
        start: int, 
        end: int, 
        raw_type: str,
        out_ptr: str,
        source: str
    ):
        """
        Adds an unpack connector to the connectors dictionary 
        based on the information passed in.
        
        :param dict connectors: The connectors dictionary
        :param str section: The name of the section
        :param str extents: The extents of the array
        :param int start: The start variable
        :param int end: The end variable
        :param str raw_type: The item's data type
        :param str in_ptr: The name of the in data pointer
        :param str out_ptr: The name of the out data pointer
        """
        offset = f"{extents} * static_cast<std::size_t>({start});"
        nBytes = f'{extents} * ( {end} - {start} + 1 ) * sizeof({raw_type});'
        data_pointer_string = cls._SOURCE_TILE_DATA_MAPPING[source.upper()]

        connectors[cls._IN_PTRS].append(
            f'{raw_type}* {out_ptr}_data_h = {data_pointer_string};\n'
        )
        connectors[f'unpack_{section}'].extend([
            f'constexpr std::size_t offset_{out_ptr} = {offset}\n',
            f'{raw_type}*        start_h_{out_ptr} = {out_ptr}_data_h + offset_{out_ptr};\n'
            f'const {raw_type}*  start_p_{out_ptr} = {out_ptr}_data_p + offset_{out_ptr};\n'
            f'constexpr std::size_t nBytes_{out_ptr} = {nBytes}\n',
            f'std::memcpy(static_cast<void*>(start_h_{out_ptr}), static_cast<const void*>(start_p_{out_ptr}), nBytes_{out_ptr});\n\n'
        ])
        # I'm the casting here is awful but I'm not sure there's a way around it that isn't just using c-style casting, 
        # and that is arguably worse than CPP style casting
        connectors[cls._OUT_PTRS].append(
            f'{raw_type}* {out_ptr}_data_p = static_cast<{raw_type}*>('
            + ' static_cast<void*>( static_cast<char*>( static_cast<void*>(' 
            + f' _{out_ptr}_p ) ) + n * SIZE_{out_ptr.upper()} ) );\n'
        )

    @classmethod
    def write_connectors(cls, connectors: dict, template):
        """
        Writes connectors to the datapacket file.
        
        :param template: The file to write the connectors to.
        :param dict connectors: The dict containing all cgkit connectors.
        """
        # constructor args requires a special formatting 
        template.writelines([
            f'/* _connector:constructor_args */\n'
            ] + [ ','.join(connectors[cls._CON_ARGS]) ] + ['\n\n'])
        del connectors[cls._CON_ARGS]

        # set members needs a special formatting
        template.writelines(
            [ f'/* _connector:set_members */\n'] + 
            [',\n'.join(connectors[cls._SET_MEMBERS]) ] + 
            ['\n\n']
        )
        del connectors[cls._SET_MEMBERS]

        template.writelines(
            [ f'/* _connector:host_members */\n' ] + 
            [','.join(connectors[cls._HOST_MEMBERS])] + 
            ['\n\n']
        )
        del connectors[cls._HOST_MEMBERS]

        # write any leftover connectors
        for connection in connectors:
            template.writelines([
                f'/* _connector:{connection} */\n'
            ] + [ ''.join(connectors[connection]) ] + ['\n'] 
        )

    @staticmethod
    def write_size_connectors(size_connectors: dict, file):
        """
        Writes the size connectors to the specified file.

        :param dict size_connectors: The dictionary of size connectors for use with CGKit.
        :param TextIO file: The file to write to.
        :rtype: None
        """
        for key,item in size_connectors.items():
            file.write(f'/* _connector:{key} */\n{item}\n\n')

    # virtual method
    @classmethod
    def insert_farray_information(cls, connectors, tile_data):
        return