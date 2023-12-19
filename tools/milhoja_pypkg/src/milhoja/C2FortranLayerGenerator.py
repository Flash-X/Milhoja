from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy

from .parse_helpers import parse_extents
from .TemplateUtility import TemplateUtility
from .FortranTemplateUtility import FortranTemplateUtility
from .AbcCodeGenerator import AbcCodeGenerator
from .TaskFunction import TaskFunction
from .LogicError import LogicError
from .constants import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG, EXTERNAL_ARGUMENT,
    TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT, GRID_DATA_ARGUMENT,
    SCRATCH_ARGUMENT, TILE_ARGUMENTS_ALL, GRID_DATA_EXTENTS
)


@dataclass
class C2FInfo:
    """
    Scruct containing various attributes of the
    pointers to be generated for the c2f layer.
    """
    name: str
    ctype: str
    ftype: str
    shape: list
    conversion_eq: str

    @property
    def dummy_shp(self):
        """Returns the dummy argument's shape string."""
        if not self.shape:
            return ""
        return '(' + ','.join([':'] * len(self.shape)) + ')'

    @property
    def cname(self):
        """Name of the dummy variable."""
        return f"C_{self.name}"

    @property
    def fname(self):
        """Name of the fortran variable."""
        return f"F_{self.name}"


class C2FortranLayerGenerator(AbcCodeGenerator):
    """
    C to Fortran layer generator. Should only be used internally with
    the DataPacketGenerator.

    Generates the C to Fortran layer for data packets.

    Note: The same JSON file used to generate the data packets
        must also be used as a parameter for this script.

        ..todo::
            * Add new tile_metadata functionality
            (cell_volumes, cell coords, etc.)
            * Add functionality for external arguments with extents.
            * Full lbound functionality?
            * Jared mentioned that this and the cpp2c layers should probably
            be separated out from the data packet generator class and moved
            over into generate_task_function.
    """
    TYPE_MAPPING = {
        'int': 'integer',
        'real': 'real',
        'bool': 'logical'
    }

    def __init__(
        self,
        tf_spec,
        indent,
        logger,
        n_ex_streams
    ):
        """
        Initializer

        :param tf_spec: The task function specification
        :param int indent: The indent size
        :param logger: The logger to be used with the class
        :param n_ex_streams: The number of extra streams
        """
        self._n_extra_streams = n_ex_streams

        # pass in an empty file for the header name since there is no header.
        super().__init__(
            tf_spec, "",
            tf_spec.output_filenames[TaskFunction.C2F_KEY]["source"],
            indent, "Milhoja C2F Generator",
            logger
        )

        self.INDENT = " " * indent

    @property
    def c2f_file(self) -> str:
        """Returns the name of the source c2f file."""
        return super().source_filename

    def generate_header_code(self, destination, overwrite):
        """No implementation for generating header code for c2f layer."""
        raise LogicError("No header file for C to Fortran layer.")

    def generate_source_code(self, destination, overwrite):
        """
        Wrapper for _generate_advance_c2f. Checks if the destination exists
        and the overwrite flag.
        """
        destination_path = Path(destination).resolve()
        if not destination_path.is_dir():
            raise RuntimeError(
                f"{destination_path} does not exist."
            )
        c2f_path = destination_path.joinpath(self.c2f_file).resolve()

        if c2f_path.is_file():
            self._warn(f"{c2f_path} already exists.")
            if not overwrite:
                self.log_and_abort(
                    f"Overwrite is {overwrite}",
                    FileExistsError()
                )
        self._generate_advance_c2f(c2f_path)

    # Note: bad source is caught by the task function so we don't need to
    #       check that.
    def __get_array_extents(self, spec) -> list:
        """
        Get the extents of an array given *spec* information. Raises specific
        errors depending on the source inside the array spec.

        :param spec: The argument spec of a given array.
        :rtype: list
        :return: The extents of the given arg spec as a list.
        """
        src = spec["source"]

        if src == EXTERNAL_ARGUMENT:
            raise NotImplementedError("Extents with externals has no tests.")

        if src in TILE_ARGUMENTS_ALL:
            raise LogicError("Tile metadata does not have array extents.")

        # todo: Consider moving this into parse helpers function.
        if src == GRID_DATA_ARGUMENT:
            struct_index = spec["structure_index"]
            block_extents = self._tf_spec.block_interior_shape
            nguard = self._tf_spec.n_guardcells
            extents = deepcopy(GRID_DATA_EXTENTS[struct_index[0]])

            for idx, ext in enumerate(block_extents):
                extents[idx] = extents[idx].format(ext, nguard)

            return extents
        elif src == SCRATCH_ARGUMENT:
            return parse_extents(spec['extents'])

        # we should never get here!
        # This might be a source allowed by the TF but is not supported.
        raise LogicError(f"Source {src} is not supported.")

    def _get_external_info(self, arg, spec) -> C2FInfo:
        """
        Convert external variable information into C2FInfo data.

        :param arg: The name of the variable.
        :param spec: The variable spec.
        """
        dtype = spec['type']
        extents = parse_extents(spec['extents'])
        use_shape = ""

        if extents:
            extents = extents + ['nTiles']
            use_shape = f", shape=[{','.join(extents)}]"

        info = C2FInfo(
            name=f'{arg}_d', ctype='type(C_PTR)',
            ftype=self.TYPE_MAPPING[dtype] + ", pointer",
            shape=extents,
            conversion_eq='CALL C_F_POINTER({0}, {1}{2})'
        )

        info.conversion_eq = info.conversion_eq.format(
            info.cname, info.fname, use_shape
        )

        return info

    def _get_metadata_info(self, arg, spec):
        """
        Convert tile metadata information to C2FInfo class using arg specs.

        :param arg: The name of the variable
        :param spec: The arg spec of the variable
        """
        assoc_array = spec.get('array', None)
        dtype = TemplateUtility.SOURCE_DATATYPE[spec["source"]]
        dtype = FortranTemplateUtility.F_HOST_EQUIVALENT[dtype]
        dtype = self.TYPE_MAPPING.get(dtype, dtype)
        shape = []
        info = C2FInfo(
            name=arg + "_d", ctype="type(C_PTR)", ftype=dtype + ", pointer",
            shape=shape,
            conversion_eq="CALL C_F_POINTER({0}, {1}, shape=[{2}])"
        )

        if assoc_array:
            array_spec = self._tf_spec.argument_specification(assoc_array)
            # add 1 for var masking
            grid_adjust = 1 if array_spec['source'] == GRID_DATA_ARGUMENT \
                else 0
            extents = self.__get_array_extents(array_spec)
            shape = [str(len(extents) + grid_adjust)] + \
                ['F_nTiles_h']
            info.shape = shape
            info.ftype = 'integer, pointer'
            info.conversion_eq = info.conversion_eq.format(
                info.cname, info.fname, ','.join(info.shape)
            )

        else:
            interior = TILE_INTERIOR_ARGUMENT
            arrayBound = TILE_ARRAY_BOUNDS_ARGUMENT
            if arg == interior or arg == arrayBound:
                info.shape = ['2', 'MILHOJA_MDIM', 'F_nTiles_h']
            else:
                info.shape = ['MILHOJA_MDIM', 'F_nTiles_h']
            info.conversion_eq = info.conversion_eq.format(
                info.cname, info.fname, ', '.join(info.shape)
            )
        return info

    def _get_grid_info(self, arg, spec):
        """
        Convert grid data information in C2FInfo class.

        :param arg: The name of the variable
        :param spec: The arg specification of the variable
        """
        # grid dtype is always real.
        dtype = "real"
        extents = self.__get_array_extents(spec)

        largest = TemplateUtility.get_array_size(
            spec.get('variables_in', None),
            spec.get('variables_out', None)
        )
        info = C2FInfo(
            name=arg + "_d", ctype="type(C_PTR)", ftype=dtype + ", pointer",
            shape=extents + [str(largest), 'F_nTiles_h'],
            conversion_eq="CALL C_F_POINTER({0}, {1}, shape=[{2}])"
        )
        info.conversion_eq = info.conversion_eq.format(
            info.cname, info.fname, ', '.join(info.shape)
        )
        return info

    def _get_scratch_info(self, arg, spec):
        """
        Convert scratch variable information into a C2FInfo class.

        :param arg: The variable name
        :param spec: The argument specification for the variable.
        """
        dtype = spec['type']
        dtype = self.TYPE_MAPPING.get(dtype, dtype)
        extents = self.__get_array_extents(spec)
        info = C2FInfo(
            name=arg + "_d", ctype="type(C_PTR)", ftype=dtype + ", pointer",
            shape=extents + ["F_nTiles_h"],
            conversion_eq="CALL C_F_POINTER({0}, {1}, shape=[{2}])"
        )
        info.conversion_eq = info.conversion_eq.format(
            info.cname, info.fname,
            ', '.join(info.shape)
        )
        return info

    def _generate_advance_c2f(self, file):
        """
        Generates the code for passing a data packet from
        the C layer to the Fortran layer to c2f.F90

        :param dict data: The json file used to generate the data
                          packet associated with this file.
        """
        self._log(f"Generating C2F layer at {str(file)}", LOG_LEVEL_BASIC)
        with open(file, 'w') as fp:
            # should size_t be translated if using fortran?
            c2f_arg_info = []
            real_args = []

            packet = C2FInfo(
                name='packet_h', ctype='type(C_PTR)', ftype='',
                shape=[], conversion_eq=''
            )
            queue_info = [
                C2FInfo(
                    name=f'queue{i}_h', ctype='integer(MILHOJA_INT)',
                    ftype='integer(kind=acc_handle_kind)', shape=[],
                    conversion_eq='{0} = INT({1}, kind=acc_handle_kind)'
                )
                for i in range(1, self._n_extra_streams+2)
            ]
            for info in queue_info:
                info.conversion_eq = info.conversion_eq.format(
                    info.fname, info.cname
                )
            # host nTiles
            nTiles = C2FInfo(
                name="nTiles_h", ctype='integer(MILHOJA_INT)',
                ftype='integer', shape=[], conversion_eq='{0} = INT({1})'
            )
            nTiles.conversion_eq = nTiles.conversion_eq.format(
                nTiles.fname, nTiles.cname
            )
            device_nTiles = C2FInfo(
                name='nTiles_d', ctype='type(C_PTR)',
                ftype='integer, pointer', shape=[],
                conversion_eq="CALL C_F_POINTER({0}, {1})"
            )
            device_nTiles.conversion_eq = device_nTiles.conversion_eq.format(
                device_nTiles.cname, device_nTiles.fname
            )

            c2f_arg_info.append(packet)
            real_args.append(packet)
            c2f_arg_info.extend(queue_info)
            real_args.extend(queue_info)
            c2f_arg_info.append(nTiles)
            c2f_arg_info.append(device_nTiles)
            real_args.append(device_nTiles)

            # move through every arg in the argument list.
            for variable in self._tf_spec.dummy_arguments:
                spec = self._tf_spec.argument_specification(variable)
                source = spec["source"]
                info = None

                if source == EXTERNAL_ARGUMENT:
                    info = self._get_external_info(variable, spec)
                elif source in TILE_ARGUMENTS_ALL:
                    info = self._get_metadata_info(variable, spec)
                elif source == GRID_DATA_ARGUMENT:
                    info = self._get_grid_info(variable, spec)
                elif source == SCRATCH_ARGUMENT:
                    info = self._get_scratch_info(variable, spec)

                if info:
                    c2f_arg_info.append(info)
                    real_args.append(info)

            # # WRITE BOILERPLATE
            fp.writelines([
                '#include "Milhoja.h"\n',
                '#ifndef MILHOJA_OPENACC_OFFLOADING\n',
                '#error "This file should only be compiled '
                'if using OpenACC offloading"\n',
                '#endif\n\n',
                f'subroutine {self._tf_spec.name}_C2F( &\n'
            ])

            # write dummy argument list
            fp.write(
                ', &\n'.join([info.cname for info in c2f_arg_info]) +
                ' &\n) bind(c)\n\n'
            )

            fortran_mod = self._tf_spec.output_filenames[
                TaskFunction.FORTRAN_TF_KEY
            ]
            fortran_mod = fortran_mod["source"]
            fortran_mod = fortran_mod[:fortran_mod.rfind(".")]
            fp.writelines([
                f'{self.INDENT}use iso_c_binding, ONLY : C_PTR, C_F_POINTER\n'
                f'{self.INDENT}use openacc, ONLY : acc_handle_kind\n',
                f'{self.INDENT}use milhoja_types_mod, ONLY : MILHOJA_INT\n',
                f'{self.INDENT}use {fortran_mod}, ONLY : '
                f'{self._tf_spec.name}_Fortran\n',
                f'{self.INDENT}implicit none\n\n'
            ])

            # write c declarations
            fp.writelines([
                f'{self.INDENT}{info.ctype}, '
                f'intent(IN), value :: {info.cname}\n'
                for info in c2f_arg_info
            ])
            fp.write('\n')

            # write f declarations
            fp.writelines([
                f'{self.INDENT}{info.ftype} :: {info.fname}{info.dummy_shp}\n'
                for info in c2f_arg_info if info.ftype
            ])

            # write c to f conversions
            fp.writelines([
                f'{self.INDENT}{info.conversion_eq}\n'
                for info in c2f_arg_info
            ])
            fp.write('\n')

            fp.write(f'{self.INDENT}CALL {self._tf_spec.name}_Fortran ( &\n')
            fp.write(
                (self.INDENT * 2) +
                f', &\n{self.INDENT * 2}'.join([
                    info.fname if info.ftype != ''
                    else info.cname for info in real_args
                ]) +
                f' &\n{self.INDENT})'
            )

            fp.write(f'\nend subroutine {self._tf_spec.name}_C2F')
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)

    def log_and_abort(self, msg, e: BaseException):
        self._error(msg)
        raise e
