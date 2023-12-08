from pathlib import Path
from dataclasses import dataclass

from .TemplateUtility import TemplateUtility
from .FortranTemplateUtility import FortranTemplateUtility
from .AbcCodeGenerator import AbcCodeGenerator
from .TaskFunction import TaskFunction
from .LogicError import LogicError
from .constants import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    # SOURCE_DATATYPE_MAPPING
)


@dataclass
class C2FInfo:
    """
    Scruct containing various attributes of the
    pointers to be generated for the c2f layer.
    """
    ftype: str
    ctype: str
    kind: str
    shape: list


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
            data_mapping = {
                'int': 'integer',
                'real': 'real',
                'bool': 'logical'
            }

            # # WRITE BOILERPLATE
            fp.writelines([
                '#include "Milhoja.h"\n',
                '#ifndef MILHOJA_OPENACC_OFFLOADING\n',
                '#error "This file should only be compiled '
                'if using OpenACC offloading"\n',
                '#endif\n\n',
                f'subroutine {self._tf_spec.name}_C2F('
            ])

            # initialize host pointers
            host_pointers = {
                'packet': C2FInfo('', 'type(C_PTR)', '', []),
                **{
                    f'queue{i}': C2FInfo(
                            'integer', 'integer(MILHOJA_INT)',
                            'acc_handle_kind', []
                        ) for i in range(1, self._n_extra_streams+2)
                  }
            }

            # get argument order and insert nTiles
            # (This affects the CPP2C generator as well.)
            arg_order = ["nTiles"] + self._tf_spec.dummy_arguments
            # ..todo:: this should probably be renamed to device pointers
            gpu_pointers = {
                'nTiles': C2FInfo('integer', 'type(C_PTR)', '', [])
            }

            # Load all items from general.
            for key, data in self._externals.items():
                ftype = data_mapping[data["type"].lower()]
                gpu_pointers[key] = C2FInfo(ftype, 'type(C_PTR)', '', [])

            # load all items from tile_metadata.
            for key, data in self._tile_metadata.items():
                ftype = FortranTemplateUtility \
                    .F_HOST_EQUIVALENT[
                        TemplateUtility.SOURCE_DATATYPE[data["source"]]
                    ].lower()
                ftype = data_mapping[ftype]
                gpu_pointers[key] = C2FInfo(
                    ftype, 'type(C_PTR)', '', ['MILHOJA_MDIM', 'F_nTiles_h']
                )

            # need to create an extents set to set the sizes for the
            # fortran arrays.
            extents_set = {
                'nTiles': C2FInfo('integer', 'integer(MILHOJA_INT)', '', [])
            }
            # load all items from array sections, except scratch.
            for section in [self._tile_in, self._tile_in_out, self._tile_out]:
                for item, data in section.items():
                    ftype = data["type"].lower()
                    ftype = data_mapping[ftype]
                    shape = ['F_nTiles_h']
                    var_in = data.get("variables_in", None)
                    var_out = data.get("variables_out", None)

                    # TODO: Fortran index space?1
                    array_size = TemplateUtility.get_array_size(
                        var_in, var_out
                    )
                    index_space = TemplateUtility.DEFAULT_INDEX_SPACE

                    shape.insert(
                        0, f'{str(array_size)} + 1 - {str(index_space)}'
                    )

                    gpu_pointers[item] = C2FInfo(
                        ftype, 'type(C_PTR)', '', data["extents"] + shape
                    )

            # finally load scratch data
            for item, data in self._scratch.items():
                ftype = data['type'].lower()
                if ftype.endswith('int'):
                    ftype = 'integer'
                    self.log_and_abort(
                        "No test cases for integer in scratch variables.",
                        NotImplementedError()
                    )
                gpu_pointers[item] = C2FInfo(
                    ftype, 'type(C_PTR)', '',
                    data["extents"] + ['F_nTiles_h']
                )

            host_pointers.update(extents_set)
            fortran_mod = self._tf_spec.output_filenames[
                TaskFunction.FORTRAN_TF_KEY
            ]
            fortran_mod = fortran_mod["source"]
            fortran_mod = fortran_mod[:fortran_mod.rfind(".")]
            # get pointers for every section
            fp.writelines([
                # put all host items into func declaration
                ', &\n'.join(f'C_{item}_h' for item in host_pointers) + \
                ', &\n',
                # we can assume every item in the TFAL exists in the
                # data packet at this point
                ', &\n'.join(f'C_{item}_d' for item in arg_order),
                ') bind(c)\n',
                f'{self.INDENT}use iso_c_binding, ONLY : C_PTR, C_F_POINTER\n'
                f'{self.INDENT}use openacc, ONLY : acc_handle_kind\n',
                f'{self.INDENT}use milhoja_types_mod, ONLY : MILHOJA_INT\n',
                f'{self.INDENT}use {fortran_mod}, ONLY : ' \
                f'{self._tf_spec.name}_Fortran\n',
                f'{self.INDENT}implicit none\n\n'
            ])

            # write c pointer & host fortran declarations
            fp.writelines([
                f'{self.INDENT}{data.ctype}, intent(IN), value :: C_{item}_h\n'
                for item, data in host_pointers.items() if data.ctype] +
                ['\n']
            )
            fp.writelines([
                f'{self.INDENT}{data.ctype}, intent(IN), value :: C_{item}_d\n'
                for item, data in gpu_pointers.items() if data.ctype] +
                ['\n']
            )
            fp.writelines([
                (
                    f'{self.INDENT}{data.ftype}'
                    f'{"" if not data.kind else f"(kind={ data.kind })"}'
                    f':: F_{item}_h\n'
                )
                for item, data in host_pointers.items() if data.ftype] +
                ['\n']
            )

            # write Fortran pointer declarations
            fp.writelines([
                f"""{self.INDENT}{data.ftype}, pointer :: F_{item}_d{''
                    if not data.shape else '(' + ','.join(
                        ':' for _ in range(0, len(data.shape))
                    ) + ')'}\n"""
                for item, data in gpu_pointers.items()] +
                ['\n']
            )

            fp.writelines([
                (
                    f"{self.INDENT}F_{item}_h = INT(C_{item}_h"
                    f"{f', kind={data.kind}' if data.kind else ''})\n"
                )
                for item, data in host_pointers.items() if data.ftype] +
                ['\n']
            )

            # remove all items in extents set so they don't get
            # passed to the task function
            for item in extents_set:
                host_pointers.pop(item)

            c2f_pointers = [
                f"""{self.INDENT}CALL C_F_POINTER(C_{item}_d, F_{item}_d{
                    f', shape=[{", ".join(ext for ext in data.shape)}]'
                    if data.shape else ''
                })\n"""
                for item, data in gpu_pointers.items() if data.ftype
            ]
            fp.writelines(c2f_pointers + ['\n'])

            # CALL STATIC FORTRAN LAYER
            fp.writelines([
                f'{self.INDENT}CALL {self._tf_spec.name}_Fortran(',
                ', &\n'.join(
                    f'{self.INDENT * 2}F_{ptr}_h'
                    if data.ftype else f'C_{ptr}_h'
                    for ptr, data in host_pointers.items()
                ),
                ', &\n',
                ', &\n'.join(
                    f'{self.INDENT * 2}F_{ptr}_d' for ptr in arg_order
                )
            ])
            fp.write(f')\nend subroutine {self._tf_spec.name}_C2F')
        self._log("Done", LOG_LEVEL_BASIC_DEBUG)

    def log_and_abort(self, msg, e: BaseException):
        self._error(msg)
        raise e
