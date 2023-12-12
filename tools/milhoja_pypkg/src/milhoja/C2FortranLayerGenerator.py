from pathlib import Path
from dataclasses import dataclass

from .parse_helpers import parse_extents
from .TemplateUtility import TemplateUtility
from .FortranTemplateUtility import FortranTemplateUtility
from .AbcCodeGenerator import AbcCodeGenerator
from .TaskFunction import TaskFunction
from .LogicError import LogicError
from .constants import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    EXTERNAL_ARGUMENT, TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_INTERIOR_ARGUMENT, TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_ARRAY_BOUNDS_ARGUMENT, GRID_DATA_ARGUMENT, SCRATCH_ARGUMENT,
    TILE_ARGUMENTS_ALL
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
    def shape_str(self):
        if not self.shape:
            return ""
        return '(' + ','.join(self.shape) + ')'

    @property
    def dummy_shp(self):
        if not self.shape:
            return ""
        return '(' + ','.join([':'] * len(self.shape)) + ')'

    @property
    def cname(self):
        return f"C_{self.name}"

    @property
    def fname(self):
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
        n_ex_streams,
        externals,
        tile_metadata,
        tile_in,
        tile_in_out,
        tile_out,
        tile_scratch
    ):
        """
        Constructor. The data packet generator automatically passes in the
        data it uses to construct the data packet classes.

        :param tf_spec: The task function specification
        :param int indent: The indent size
        :param logger: The logger to be used with the class
        :param n_ex_streams: The number of extra streams
        :param externals: All external vars
        :param tile_metadata: All tile_metadata vars
        :param tile_in: All tile_in vars
        :param tile_in_out: All tile_in_out vars
        :param tile_out: All tile_out vars
        :param tile_scratch: All tile_scratch args.
        """
        self._n_extra_streams = n_ex_streams
        self._externals = externals
        self._tile_metadata = tile_metadata
        self._tile_in = tile_in
        self._tile_in_out = tile_in_out
        self._tile_out = tile_out
        self._scratch = tile_scratch

        # pass in an empty file for the header name since there is no header.
        super().__init__(
            tf_spec, "",
            tf_spec.output_filenames[TaskFunction.C2F_KEY]["source"],
            indent, "Milhoja C2F",
            logger
        )

        self.INDENT = " " * indent

    @property
    def c2f_file(self) -> str:
        return super().source_filename

    def generate_header_code(self, destination, overwrite):
        """No implementation for generating header code for c2f layer."""
        raise LogicError(
            "No header file for C to Fortran layer. Please contact your state"
            " provided Wesley to solve this issue."
        )

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

    def _get_external_info(self, arg, spec) -> C2FInfo:
        dtype=spec['type']
        extents = parse_extents(spec['extents'])
        use_shape = ""
        if extents:
            extents = extents + ['nTiles']
            use_shape = f", shape=[{','.join(extents)}]"

        info = C2FInfo(
            name=arg, ctype='type(C_PTR)', ftype=self.TYPE_MAPPING[dtype],
            shape=extents,
            conversion_eq='CALL C_F_POINTER({0}, {1}{2})'
        )

        info.conversion_eq = info.conversion_eq.format(
            info.cname, info.fname, use_shape
        )

        return info

    def _get_metadata_info(self, arg, spec):
        ...

    def _get_grid_info(self, arg, spec):
        ...

    def _get_scratch_info(self, arg, spec):
        dtype = spec['type']
        dtype = self.TYPE_MAPPING.get(dtype, dtype)
        info = C2FInfo(
            name=arg, ctype="type(C_PTR)", ftype=dtype,
            shape=parse_extents(spec['extents']),
            conversion_eq="CALL C_F_POINTER({0}, {1}, shape=[{2}])"
        )
        info.conversion_eq = info.conversion_eq.format(
            info.cname, info.fname,
            ','.join(info.shape + ["F_nTiles_h"])
        )
        return info

    def _generate_advance_c2f(self, file):
        """
        Generates the code for passing a data packet from
        the C layer to the Fortran layer to c2f.F90

        :param dict data: The json file used to generate the data
                          packet associated with this file.
        """
        self._log("Generating c2f layer at {str(file)}", LOG_LEVEL_BASIC)
        with open(file, 'w') as fp:
            # should size_t be translated if using fortran?

            c2f_arg_info = []
            func_dummy_args = []
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

            c2f_arg_info.append(packet)
            real_args.append(packet)
            c2f_arg_info.extend(queue_info)
            real_args.extend(queue_info)
            c2f_arg_info.append(nTiles)

            # move through every arg in the argument list.
            for variable in ['nTiles'] + self._tf_spec.dummy_arguments:
                try:
                    spec = self._tf_spec.argument_specification(variable)
                    source = spec["source"]
                    info = None

                    if source == EXTERNAL_ARGUMENT:
                        info = self._get_external_info(variable, spec)
                    elif source in TILE_ARGUMENTS_ALL:
                        info = self._get_metadata_info(variable, spec)
                    elif source == GRID_DATA_ARGUMENT:
                        ...
                    elif source == SCRATCH_ARGUMENT:
                        info = self._get_scratch_info(variable, spec)

                    if info:
                        c2f_arg_info.append(info)
                        real_args.append(info)

                except: # nTiles or missing
                    print(variable + " = bad")

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
                f'{self.INDENT}use {fortran_mod}, ONLY : ' \
                f'{self._tf_spec.name}_Fortran\n',
                f'{self.INDENT}implicit none\n\n'
            ])

            # write c declarations
            fp.writelines([
                f'{self.INDENT}{info.ctype}, intent(IN), value :: {info.cname}\n'
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
                f', &\n{self.INDENT * 2}'.join([info.fname if info.fname else info.cname for info in real_args ]) +
                f' &\n{self.INDENT})'
            )


            # # get argument order and insert nTiles
            # arg_order = ["nTiles"] + self._tf_spec.dummy_arguments
            # # ..todo:: this should probably be renamed to device pointers
            # gpu_pointers = {
            #     'nTiles': C2FInfo('integer', 'type(C_PTR)', '', [])
            # }

            # # Load all items from general.
            # for key, data in self._externals.items():
            #     ftype = data_mapping[data["type"].lower()]
            #     gpu_pointers[key] = C2FInfo(ftype, 'type(C_PTR)', '', [])

            # # load all items from tile_metadata.
            # for key, data in self._tile_metadata.items():
            #     ftype = FortranTemplateUtility \
            #         .F_HOST_EQUIVALENT[
            #             TemplateUtility.SOURCE_DATATYPE[data["source"]]
            #         ].lower()
            #     ftype = data_mapping[ftype]
            #     gpu_pointers[key] = C2FInfo(
            #         ftype, 'type(C_PTR)', '', ['MILHOJA_MDIM', 'F_nTiles_h']
            #     )

            # # need to create an extents set to set the sizes for the
            # # fortran arrays.
            # extents_set = {
            #     'nTiles': C2FInfo('integer', 'integer(MILHOJA_INT)', '', [])
            # }
            # # load all items from array sections, except scratch.
            # for section in [self._tile_in, self._tile_in_out, self._tile_out]:
            #     for item, data in section.items():
            #         ftype = data["type"].lower()
            #         ftype = data_mapping[ftype]
            #         shape = ['F_nTiles_h']
            #         var_in = data.get("variables_in", None)
            #         var_out = data.get("variables_out", None)

            #         # TODO: Fortran index space?1
            #         array_size = TemplateUtility.get_array_size(
            #             var_in, var_out
            #         )
            #         index_space = TemplateUtility.DEFAULT_INDEX_SPACE

            #         shape.insert(
            #             0, f'{str(array_size)} + 1 - {str(index_space)}'
            #         )

            #         gpu_pointers[item] = C2FInfo(
            #             ftype, 'type(C_PTR)', '', data["extents"] + shape
            #         )

            # # finally load scratch data
            # for item, data in self._scratch.items():
            #     ftype = data['type'].lower()
            #     if ftype.endswith('int'):
            #         ftype = 'integer'
            #         self.log_and_abort(
            #             "No test cases for integer in scratch variables.",
            #             NotImplementedError()
            #         )
            #     gpu_pointers[item] = C2FInfo(
            #         ftype, 'type(C_PTR)', '',
            #         data["extents"] + ['F_nTiles_h']
            #     )

            # host_pointers.update(extents_set)
            

            # # write c pointer & host fortran declarations
            # fp.writelines([
            #     f'{self.INDENT}{data.ctype}, intent(IN), value :: C_{item}_h\n'
            #     for item, data in host_pointers.items() if data.ctype] +
            #     ['\n']
            # )
            # fp.writelines([
            #     f'{self.INDENT}{data.ctype}, intent(IN), value :: C_{item}_d\n'
            #     for item, data in gpu_pointers.items() if data.ctype] +
            #     ['\n']
            # )
            # fp.writelines([
            #     (
            #         f'{self.INDENT}{data.ftype}'
            #         f'{"" if not data.kind else f"(kind={ data.kind })"}'
            #         f':: F_{item}_h\n'
            #     )
            #     for item, data in host_pointers.items() if data.ftype] +
            #     ['\n']
            # )

            # # write Fortran pointer declarations
            # fp.writelines([
            #     f"""{self.INDENT}{data.ftype}, pointer :: F_{item}_d{''
            #         if not data.shape else '(' + ','.join(
            #             ':' for _ in range(0, len(data.shape))
            #         ) + ')'}\n"""
            #     for item, data in gpu_pointers.items()] +
            #     ['\n']
            # )

            # fp.writelines([
            #     (
            #         f"{self.INDENT}F_{item}_h = INT(C_{item}_h"
            #         f"{f', kind={data.kind}' if data.kind else ''})\n"
            #     )
            #     for item, data in host_pointers.items() if data.ftype] +
            #     ['\n']
            # )

            # # remove all items in extents set so they don't get
            # # passed to the task function
            # for item in extents_set:
            #     host_pointers.pop(item)

            # c2f_pointers = [
            #     f"""{self.INDENT}CALL C_F_POINTER(C_{item}_d, F_{item}_d{
            #         f', shape=[{", ".join(ext for ext in data.shape)}]'
            #         if data.shape else ''
            #     })\n"""
            #     for item, data in gpu_pointers.items() if data.ftype
            # ]
            # fp.writelines(c2f_pointers + ['\n'])

            fp.write(f'\nend subroutine {self._tf_spec.name}_C2F')
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)

    def log_and_abort(self, msg, e: BaseException):
        self._error(msg)
        raise e
