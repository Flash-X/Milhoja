"""
This is the main script for generating a new DataPacket. The overall structure
of this code is to move through every possible DataPacket JSON section and
fill out the various CGKkit dictionaries for generating every template.

..todo::
    * How to sort bound class members by size?
    * Eventually logging should be more informative and replace
      all print statements.
"""
from collections import OrderedDict
from abc import abstractmethod

from .DataPacketMemberVars import DataPacketMemberVars
from .LogicError import LogicError
from .TaskFunction import TaskFunction
from . import GRID_DATA_PTRS


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
    _TILE_DESC = "tileDesc_h"

    # C++ Index space is always 0.
    # todo:: Should this be handled by the TF Spec?
    DEFAULT_INDEX_SPACE = 0

    def __init__(self, tf_spec: TaskFunction):
        """
        Initializer for the base template utility class.
        TemplateUtility needs a copy of tf_spec in order to handle lbounds
        for any array.
        """
        self.tf_spec = tf_spec

    @classmethod
    @abstractmethod
    def iterate_externals(
        cls, connectors: dict, size_connectors: dict, externals: OrderedDict,
        dummy_arg_list: list
    ):
        ...

    @classmethod
    def _common_iterate_externals(
        cls, connectors: dict, externals: OrderedDict, dummy_arg_list: list
    ):
        """
        Common code in both utility classes for iterating external vars.

        todo::
            * Use tf_spec.dummy_arguments directly instead of dummy_arg_list.

        :param dict connectors: All cgkit connectors.
        :param dict size_connectors: All size_connectors for cgkit.
        :param OrderedDict externals: All external variables from the TF.
        :param list dummy_arg_list: The tf spec's argument list.
        """
        info_list = {}
        # MOVE THROUGH EVERY EXTERNAL ITEM
        for key, var_data in externals.items():
            size_equation = f'sizeof({var_data["type"]})'
            if var_data["extents"] != "()":
                size_equation = \
                    f'{size_equation} * {" * ".join(var_data["extents"])}'
                raise NotImplementedError(
                    "No test cases for external var with extents."
                )
            info = DataPacketMemberVars(
                item=key, dtype=var_data["type"],
                size_eq=size_equation, per_tile=False
            )
            info_list[key] = info

            # add the necessary connectors for the constructor section.
            connectors[cls._PUB_MEMBERS].extend([
                f'{info.dtype} {info.host};\n',
                f'{info.dtype}* {info.device};\n'
            ])

            set_host = f'{{{key}}}'
            # NOTE: nTiles must be set when pack is called because tiles_
            #       has not been filled at time of packet construction.
            if key == "nTiles":
                set_host = '{0}'

            connectors[cls._SET_MEMBERS].extend([
                f'{info.host}{set_host}',
                f'{info.device}{{nullptr}}'
            ])
            connectors[cls._SIZE_DET].append(
                f'static constexpr std::size_t {info.size} = '
                f'{info.SIZE_EQ};\n'
            )
            cls.set_pointer_determination(connectors, cls._EXT, info)
            connectors[f'memcpy_{cls._EXT}'].append(
                f'std::memcpy({info.pinned}, static_cast<void*>(&'
                f'{info.host}), {info.size});\n'
            )

        # sort host members and constructor args based on the task function
        # ordering.
        external_arg_list = [
            item for item in dummy_arg_list if item in externals
        ]
        for item in external_arg_list:
            info = info_list[item]
            connectors[cls._CON_ARGS].append(f'{info.dtype} {item}')
            connectors[cls._HOST_MEMBERS].append(info.host)

    @classmethod
    @abstractmethod
    def iterate_tile_metadata(
        cls, connectors: dict, size_connectors: dict,
        tilemetadata: OrderedDict, num_arrays: int
    ):
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
    def _common_iterate_tile_data(
        cls, data, connectors, info, extents, mask_in, mask_out, arg_type
    ):
        """
        Common code for iterating over tile data (tile_in, in_out, out).

        :param data: The arguments for the variable.
        :param connectors: The connectors dictionary.
        :param info: The information struct for the tile variable.
        :param extents: The size extents array for the variable.
        :param mask_in: The variable mask for packing, if it exists.
        :param mask_out: The variable mask for unpacking, if it exists.
        :param arg_type: The type of tile data (in, in_out, out).
        """
        connectors[cls._PUB_MEMBERS].append(
            f'{info.dtype}* {info.device};\n'
            f'{info.dtype}* {info.pinned};\n'
        )
        connectors[cls._SET_MEMBERS].extend(
            [
                f'{info.device}{{nullptr}}',
                f'{info.pinned}{{nullptr}}'
            ]
        )
        connectors[cls._SIZE_DET].append(
            f'static constexpr std::size_t {info.size} = '
            f'{info.SIZE_EQ};\n'
        )
        cls.set_pointer_determination(
            connectors, arg_type, info, True
        )
        if mask_in:
            cls.add_memcpy_connector(
                connectors, arg_type,
                extents, info.ITEM, mask_in[0],
                mask_in[1], info.size, info.dtype,
                data['structure_index'][0]
            )
        # here we pass in item twice because tile_in_out pointers get
        # packed and unpacked from the same location.
        if mask_out:
            cls.add_unpack_connector(
                connectors, arg_type,
                extents, mask_out[0], mask_out[1],
                info.dtype, info.ITEM,
                data['structure_index'][0]
            )

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

    @classmethod
    def _common_iterate_tile_scratch(cls, connectors, info):
        """
        Common code pulled out of each template utility's iterate_scratch
        function.

        :param dict connectors: Dict containing all cgkit connectors
        :param DataPacketMemberVars info: Data for a specific variable in the
                                          data packet.
        """
        connectors[cls._PUB_MEMBERS].append(
            f'{info.dtype}* {info.device};\n'
        )
        connectors[cls._SET_MEMBERS].append(
            f'{info.device}{{nullptr}}'
        )
        connectors[cls._SIZE_DET].append(
            f'static constexpr std::size_t {info.size} = '
            f'{info.SIZE_EQ};\n'
        )
        connectors[f'pointers_{cls._T_SCRATCH}'].append(
            f"{info.device} = static_cast<{info.dtype}*>( "
            f"static_cast<void*>(ptr_d) );\n"
            f"ptr_d += {info.total_size};\n\n"
        )

    @staticmethod
    def add_size_parameter(name: str, section_dict: dict, connectors: dict):
        """
        Adds a size connector to the params dict that is passed in. The size
        connector dictionary stores every size of every item in the data
        packet for use with a CGKit template.

        :param str name: The name of the size connector. Usually the
                         name of a section.
        :param dict section_dict: The section used to generate sizes.
                                  Contains all of the items in the section.
        :param dict connectors: The dictionary containing all connectors
                                generated for the packet.
        :rtype: None
        """
        size = '0'
        if section_dict:
            size = \
                '\n + '.join(f'SIZE_{item.upper()}' for item in section_dict)
        connectors[f'size_{name}'] = size

    @staticmethod
    def section_creation(
        name: str, section: OrderedDict, connectors: dict, size_connectors
    ):
        """
        Creates a section and sets the default value to an empty list.
        It's assumed that the function that calls this method
        will populate the dictionary using the same name.

        :param str name: The name of the section to create.
        :param dict section: The dictionary to get all data packet items from.
        :param dict connectors: The dictionary containing all link connectors.
        :param dict size_connectors: The dictionary containing all connectors
                                     that determine sizes for each variable in
                                     the data packet.
        """
        TemplateUtility.add_size_parameter(name, section, size_connectors)
        connectors[f'pointers_{name}'] = []
        connectors[f'memcpy_{name}'] = []

    @staticmethod
    def set_pointer_determination(
        connectors: dict, section: str, info: DataPacketMemberVars,
        item_is_member_variable=False
    ):
        """
        Stores the code for determining pointer offsets in the
        DataPacket into the connectors dictionary to be used with CGKit.

        :param dict connectors: The dict containing all connectors needed for
                                use with CGKit.
        :param str section: The section to use to determine the key for
                            the pointer determination.
        :param DataPacketMemberVars info: Contains information for formatting
                                          the name to get variable names.
        :param bool item_is_member_variable: Flag if *item* pinned memory
                                             pointer is a member variable.
                                             default = False.
        """
        dtype = info.dtype
        dtype += "* "
        if item_is_member_variable:
            dtype = ""
        # If the item is a data packet member variable, we don't need
        # to specify a type name here.
        # Note that pointers to memory in the remote device are always
        # member variables, so it will never need a type name.
        # if item_is_member_variable:
        #     dtype = ""
        # else:
        #     dtype += "* "

        # insert items into boiler plate for the pointer determination
        # phase for *section*.
        connectors[f'pointers_{section}'].append(
            f"{dtype}{info.pinned} = static_cast<{info.dtype}*>( "
            "static_cast<void*>(ptr_p) );\n"
            f"{info.device} = static_cast<{info.dtype}*>( "
            "static_cast<void*>(ptr_d) );\n" +
            f"ptr_p+={info.total_size};\n" +
            f"ptr_d+={info.total_size};\n\n"
        )

    @classmethod
    def generate_extra_streams_information(
        cls, connectors: dict, extra_streams: int
    ):
        """
        Fills the links extra_streams, stream_functions_h/cxx.
        TODO: Streams in the datapacket should use an array implementation

        :param dict connectors: The dictionary containing all connectors to
                                be used with CGKit.
        :param int extra_streams: The number of extra streams specified in
                                  the data packet JSON.
        :rtype: None
        """
        # throw an error if we have a negative amount of streams.
        if extra_streams < 1:
            if extra_streams < 0:
                raise LogicError("N streams should not be negative!")
            return

        connectors[cls._STREAM_FUNCS_H].extend([
            'int extraAsynchronousQueue(const unsigned int id) override;\n',
            'void releaseExtraQueue(const unsigned int id) override;\n'
        ])
        connectors[cls._EXTRA_STREAMS].extend([
            f'milhoja::Stream stream{i}_;\n'
            for i in range(2, extra_streams+2)
        ])
        connectors[cls._DESTRUCTOR].extend([
            f'if (stream{i}_.isValid())\n' +
            '\tthrow std::logic_error("'
            '[_param:class_name::~_param:class_name] ' +
            f'Stream {i} not released");\n'
            for i in range(2, extra_streams+2)
        ])

        # these insert the various stream functions if there are more
        # than 1 stream
        # normally these would be in the template but these functions
        # have no reason to exist if the
        # task function does not use more than 1 stream.
        connectors[cls._STREAM_FUNCS_CXX].extend([
                'int _param:class_name::extraAsynchronousQueue('
                'const unsigned int id) {\n',
                '\tif (id > INT_MAX)\n\t\tthrow std::overflow_error("'
                '[_param:class_name extraAsynchronousQueue]'
                ' id is too large for int.");\n'
                f'\tif((id < 2) || (id > {extra_streams} + 1))\n' +
                '\t\tthrow std::invalid_argument("'
                '[_param:class_name::extraAsynchronousQueue] '
                'Invalid id.");\n\t',
            ] + [
                f'else if (id == {i}) {{\n' +
                f'\t\tif (!stream{i}_.isValid()) ' +
                '\n\t\t\tthrow std::logic_error("'
                '[_param:class_name::extraAsynchronousQueue] ' +
                f'Stream {i} invalid.");'
                f'\n\t\treturn stream{i}_.accAsyncQueue;\n' +
                '\t} '
                for i in range(2, extra_streams+2)
            ] + [
                '\n\treturn 0;\n',
                '}\n\n'
            ] + [
                'void _param:class_name::releaseExtraQueue'
                '(const unsigned int id) {\n',
                f'\tif((id < 2) || (id > {extra_streams} + 1))\n' +
                '\t\tthrow std::invalid_argument("['
                '_param:class_name::releaseExtraQueue] Invalid id.");\n\t',
            ] + [
                f'else if(id == {i}) {{\n'
                f'\t\tif(!stream{i}_.isValid())\n' +
                '\t\t\tthrow std::logic_error("'
                '[_param:class_name::releaseExtraQueue] '
                f'Stream {i} invalid.");\n' +
                '\t\tmilhoja::RuntimeBackend::instance()'
                f'.releaseStream(stream{i}_);\n'
                '\t} '
                for i in range(2, extra_streams+2)
            ] +
            ['\n''}\n']
        )

        # Inserts the code necessary to acquire extra streams.
        connectors[cls._EXTRA_STREAMS_PACK].extend([
            f'stream{i}_ = RuntimeBackend::instance()'
            '.requestStream(true);\n' +
            f'if(!stream{i}_.isValid())\n' +
            f'\tthrow std::runtime_error("[_param:class_name::pack] '
            f'Unable to acquire stream {i}.");\n'
            for i in range(2, extra_streams+2)
        ])

    @classmethod
    def add_memcpy_connector(
        cls, connectors: dict, section: str, extents: str, item: str,
        start: int, end: int, size_item: str, raw_type: str, source: str
    ):
        """
        Adds a memcpy connector based on the information passed in.

        :param dict connectors: The dict containing all cgkit connectors
        :param str section: The section to add a memcpy connector for.
        :param str extents: The string containing array extents information.
        :param str item: The item to copy into pinned memory.
        :param int start: The starting index of the array.
        :param int end: The ending index of the array.
        :param str size_item: The string containing the size var for item.
        :param str raw_type: The data type of the item.
        :param str source: The source type of the variable
        """
        offset = f"{extents} * static_cast<std::size_t>({start})"
        nBytes = f'{extents} * ( {end} - {start} + 1 ) * sizeof({raw_type})'

        # This exists inside the pack function for copying data
        # from tile_in to the device.
        # Luckily we don't really need to use DataPacketMemberVars here
        # because the temporary device pointer is locally scoped.

        desc = cls._TILE_DESC
        data_pointer_string = GRID_DATA_PTRS[source.upper()].format(desc)

        # TODO: Use grid data to get data pointer information.
        connectors[f'memcpy_{section}'].extend([
            f'{raw_type}* {item}_d = {data_pointer_string};\n'
            f'constexpr std::size_t offset_{item} = {offset};\n',
            f'constexpr std::size_t nBytes_{item} = {nBytes};\n',
            f'char_ptr = static_cast<char*>( static_cast<void*>(_{item}_p) )'
            f' + n * {size_item};\n',
            f'std::memcpy(static_cast<void*>(char_ptr), '
            f'static_cast<void*>({item}_d + offset_{item}),'
            f' nBytes_{item});\n\n'
        ])

    @classmethod
    def add_unpack_connector(
        cls, connectors: dict, section: str, extents, start: int, end: int,
        raw_type: str, out_ptr: str, source: str
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
        :param str source: The source type of the variable
        """
        offset = f"{extents} * static_cast<std::size_t>({start});"
        nBytes = f'{extents} * ( {end} - {start} + 1 ) * sizeof({raw_type});'

        desc = cls._TILE_DESC
        data_pointer_string = GRID_DATA_PTRS[source.upper()].format(desc)

        connectors[cls._IN_PTRS].append(
            f'{raw_type}* {out_ptr}_data_h = {data_pointer_string};\n'
        )
        connectors[f'unpack_{section}'].extend([
            f'constexpr std::size_t offset_{out_ptr} = {offset}\n',
            f'{raw_type}*        '
            f'start_h_{out_ptr} = {out_ptr}_data_h + offset_{out_ptr};\n'
            f'const {raw_type}*  '
            f'start_p_{out_ptr} = {out_ptr}_data_p + offset_{out_ptr};\n'
            f'constexpr std::size_t nBytes_{out_ptr} = {nBytes}\n',
            f'std::memcpy(static_cast<void*>(start_h_{out_ptr}), '
            f'static_cast<const void*>(start_p_{out_ptr}), '
            f'nBytes_{out_ptr});\n\n'
        ])
        # I'm the casting here is awful but I'm not sure
        # there's a way around it that isn't just using c-style casting,
        # and that is arguably worse than C++ style casting
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
        template.writelines(
            ['/* _connector:constructor_args */\n'] +
            [','.join(connectors[cls._CON_ARGS])] +
            ['\n\n']
        )
        del connectors[cls._CON_ARGS]

        # set members needs a special formatting
        template.writelines(
            ['/* _connector:set_members */\n'] +
            [',\n'.join(connectors[cls._SET_MEMBERS])] +
            ['\n\n']
        )
        del connectors[cls._SET_MEMBERS]

        template.writelines(
            ['/* _connector:host_members */\n'] +
            [','.join(connectors[cls._HOST_MEMBERS])] +
            ['\n\n']
        )
        del connectors[cls._HOST_MEMBERS]

        # write any leftover connectors
        for connection in connectors:
            template.writelines(
                [f'/* _connector:{connection} */\n'] +
                [''.join(connectors[connection])] +
                ['\n']
            )

    @staticmethod
    def write_size_connectors(size_connectors: dict, file):
        """
        Writes the size connectors to the specified file.

        :param dict size_connectors: The dictionary of size
                                     connectors for use with CGKit.
        :param TextIO file: The file to write to.
        """
        for key, item in size_connectors.items():
            file.write(f'/* _connector:{key} */\n{item}\n\n')

    # virtual method
    @classmethod
    def insert_farray_information(cls, connectors, tile_data):
        return
