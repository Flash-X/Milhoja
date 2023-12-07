import pathlib

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

from . import AbcCodeGenerator
from . import LogicError
from . import TaskFunction
from .parse_helpers import parse_lbound_f
from .parse_helpers import parse_extents
from . import GRID_DATA_FUNC_MAPPING
from . import SOURCE_DATATYPE_MAPPING
from . import C2F_TYPE_MAPPING
from .TemplateUtility import TemplateUtility
from . import (
    EXTERNAL_ARGUMENT,
    TILE_LO_ARGUMENT,
    TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT,
    TILE_UBOUND_ARGUMENT,
    F_HOST_EQUIVALENT,
    THREAD_INDEX_VAR_NAME,
    GRID_DATA_ARGUMENT,
    SCRATCH_ARGUMENT,
    TILE_INTERIOR_ARGUMENT,
    TILE_ARRAY_BOUNDS_ARGUMENT
)


@dataclass
class ConversionData:
    """
    Scruct containing various attributes of the
    pointers to be generated for the c2f layer.
    """
    cname: str
    fname: str
    dtype: str
    is_pointer: bool
    shape: list


class TaskFunctionC2FGenerator_cpu_F(AbcCodeGenerator):
    """
    Generates the Cpp2C layer for a TaskFunction using a CPU data item
    with a fortran based task function.
    """
    C2F_ARG_LIST = "c2f_dummy_args"
    TILE_DATA = "tile_data"
    AC_SCRATCH = "acquire_scratch"
    CONSOLIDATE_TILE_DATA = "consolidate_tile_data"
    REAL_ARGS = "real_args"

    def __init__(self, tf_spec: TaskFunction, indent, logger):
        header = None
        source = \
            tf_spec.output_filenames[TaskFunction.C2F_KEY]["source"]
        self.INDENT = ' ' * indent

        self.stree_opts  = {
            'codePath': pathlib.Path.cwd(),
            'indentSpace': ' ' * indent,
            'verbose': False,
            'verbosePre': '/* ',
            'verbosePost': ' */',
        }

        super().__init__(
            tf_spec, header, source, indent,
            "Milhoja TF Cpp2C", logger
        )

    def _get_external_info(self, arg, arg_spec) -> ConversionData:
        dtype = C2F_TYPE_MAPPING[arg_spec["type"]]
        shape = \
            parse_extents(arg_spec["extents"]) if arg_spec["extents"] else []
        return ConversionData(
            cname=f"C_{arg}",
            fname=f"F_{arg}",
            dtype=dtype,
            is_pointer=True if shape else False,
            shape=shape
        )

    def _get_tmdata_info(self, arg, arg_spec, saved=set()) \
    -> Optional[ConversionData]:
        source = arg_spec["source"]
        assoc_array = arg_spec.get("array", None)

        # array lbound argument.
        if assoc_array:
            array_arg_spec = self._tf_spec.argument_specification(assoc_array)
            lbound = array_arg_spec.get("lbound", None)
            lbound = lbound if lbound else "(tile_lbound, 1)"
            bound_size = len(parse_lbound_f(lbound))
            return ConversionData(
                cname=f"C_{arg}",
                fname=f"F_{arg}",
                dtype="integer",
                is_pointer=True,
                shape=[str(bound_size)]
            )
        # otherwise normal mdata argument
        else:
            combine_bounds = self._tf_spec.use_combined_array_bounds
            if combine_bounds[0] and \
            (source == TILE_LO_ARGUMENT or source == TILE_HI_ARGUMENT):
                if TILE_INTERIOR_ARGUMENT in saved:
                    return None
                saved.add(TILE_INTERIOR_ARGUMENT)
                return ConversionData(
                    cname=f"C_{TILE_INTERIOR_ARGUMENT}",
                    fname=f"F_{TILE_INTERIOR_ARGUMENT}",
                    dtype="integer",
                    is_pointer=True,
                    shape=["2", "MILHOJA_MDIM"]
                )
            elif combine_bounds[1] and \
            (source == TILE_LBOUND_ARGUMENT or source == TILE_UBOUND_ARGUMENT):
                if TILE_ARRAY_BOUNDS_ARGUMENT in saved:
                    return None
                saved.add(TILE_ARRAY_BOUNDS_ARGUMENT)
                return ConversionData(
                    cname=f"C_{TILE_ARRAY_BOUNDS_ARGUMENT}",
                    fname=f"F_{TILE_ARRAY_BOUNDS_ARGUMENT}",
                    dtype="integer",
                    is_pointer=True,
                    shape=["2", "MILHOJA_MDIM"]
                )
            else:
                dtype = F_HOST_EQUIVALENT[SOURCE_DATATYPE_MAPPING[source]]
                dtype = C2F_TYPE_MAPPING[dtype]
                return ConversionData(
                    cname=f"C_{arg}",
                    fname=f"F_{arg}",
                    dtype=dtype,
                    is_pointer=True,
                    shape=["MILHOJA_MDIM"]
                )

    def _get_grid_info(self, arg, arg_spec) -> ConversionData:
        # HELL
        dtype = SOURCE_DATATYPE_MAPPING[arg_spec["source"]]
        dtype = C2F_TYPE_MAPPING[dtype]

        mask_in = arg_spec.get("variables_in", [])
        mask_out = arg_spec.get("variables_out", [])
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        st_index = arg_spec["structure_index"][0]
        block_size = self._tf_spec.block_interior_shape
        x = '({0}) + 1' if st_index.lower() == "fluxx" else '{0}'
        y = '({0}) + 1' if st_index.lower() == "fluxy" else '{0}'
        z = '({0}) + 1' if st_index.lower() == "fluxz" else '{0}'

        gcells = self._tf_spec.n_guardcells
        shape = [
            x.format(f'{block_size[0]} + 2 * {gcells} * MILHOJA_K1D'),
            y.format(f'{block_size[1]} + 2 * {gcells} * MILHOJA_K2D'),
            z.format(f'{block_size[2]} + 2 * {gcells} * MILHOJA_K3D'),
            str(size)
        ]
        return ConversionData(
            cname=f"C_{arg}",
            fname=f"F_{arg}",
            dtype=dtype,
            is_pointer=True,
            shape=shape
        )

    def _get_scratch_info(self, arg, arg_spec) -> ConversionData:
        dtype = C2F_TYPE_MAPPING[arg_spec["type"]]
        shape = parse_extents(arg_spec["extents"])
        return ConversionData(
            cname=f"C_{arg}",
            fname=f"F_{arg}",
            dtype=dtype,
            is_pointer=True,
            shape=shape
        )

    def _generate_c2f(self, destination: Path, overwrite):
        c2f_file = destination.joinpath(self.source_filename).resolve()

        if c2f_file.is_file():
            self._warn(f"{str(c2f_file)} already exists.")
            if overwrite:
                self._error("Overwrite is set to False.")
                raise FileExistsError()

        with open(c2f_file, 'w') as c2f:
            # get external vars
            func_args = []
            arg_conversion_info = []

            arg_list = self._tf_spec.dummy_arguments
            for arg in arg_list:
                arg_spec = self._tf_spec.argument_specification(arg)
                conversion_info = None
                if arg_spec["source"] == EXTERNAL_ARGUMENT:
                    conversion_info = self._get_external_info(arg, arg_spec)
                elif arg_spec["source"].startswith("tile_"):
                    conversion_info = self._get_tmdata_info(arg, arg_spec)
                elif arg_spec["source"] == GRID_DATA_ARGUMENT:
                    conversion_info = self._get_grid_info(arg, arg_spec)
                elif arg_spec["source"] == SCRATCH_ARGUMENT:
                    conversion_info = self._get_scratch_info(arg, arg_spec)

                if conversion_info:
                    arg_conversion_info.append(conversion_info)

            routine_name = self._tf_spec.c2f_layer_name
            c2f.write(f'#include "Milhoja.h"\n\n')
            c2f.write(f'subroutine {routine_name}( &\n')
            arg_list = f', &\n{self.INDENT}'.join(func_args)
            arg_list = f', &\n{self.INDENT}'.join([
                info.cname for info in arg_conversion_info
            ])
            c2f.write(f'{self.INDENT}{arg_list} &\n) bind(c)\n')

            c2f.write(
                f'{self.INDENT}use iso_c_binding, ONLY : C_PTR, C_F_POINTER\n'
            )
            c2f.write(
                f'{self.INDENT}use milhoja_types_mod, '
                'ONLY : MILHOJA_INT, MILHOJA_REAL\n'
            )
            f_name = self._tf_spec.fortran_module_name
            tf_name = self._tf_spec.name + "_Fortran"
            c2f.write(f'{self.INDENT}use {f_name}, ONLY : {tf_name}\n')
            c2f.write(f'{self.INDENT}implicit none\n\n')

            # write c var declarations
            c_declarations = []
            f_declarations = []
            for info in arg_conversion_info:
                cdtype = info.dtype
                if info.is_pointer:
                    cdtype = "type(C_PTR)"
                elif cdtype == "real":
                    cdtype = "real(MILHOJA_REAL)"
                elif cdtype == "integer":
                    cdtype = "integer(MILHOJA_INT)"

                intent = "intent(IN)"
                value = info.cname
                c_declarations.append(
                    f"{self.INDENT}{cdtype}, {intent}, value :: {value}"
                )
            c2f.write('\n'.join(c_declarations) + "\n\n")

            # write f var declarations
            for info in arg_conversion_info:
                fdtype = info.dtype
                if info.is_pointer:
                    fdtype += ", pointer"
                shape = [":"] * len(info.shape)
                shape = ','.join(shape)
                if shape:
                    shape = f'({shape})'
                f_declarations.append(
                    f"{self.INDENT}{fdtype} :: {info.fname}{shape}"
                )
            c2f.write('\n'.join(f_declarations) + "\n\n")

            # set values
            set_values = []
            for info in arg_conversion_info:
                if not info.is_pointer:
                    dtype = info.dtype if info.dtype != "integer" else "int"
                    set_values.append(
                        f"{self.INDENT}{info.fname} = " \
                        f"{dtype.upper()}({info.cname})"
                    )
                else:
                    set_values.append(
                        f"{self.INDENT}CALL C_F_POINTER({info.cname}, " \
                        f"{info.fname}, shape=[{','.join(info.shape)}])"
                    )
            c2f.write('\n'.join(set_values) + "\n\n")

            # call task function with new vars
            tf_real_args = f', &\n{self.INDENT * 2}'.join([
                info.fname for info in arg_conversion_info
            ])
            c2f.write(f'{self.INDENT}CALL {tf_name}( &\n{self.INDENT * 2}')
            c2f.write(tf_real_args + ")\n")
            c2f.write(f"end subroutine {routine_name}\n")

    def generate_header_code(self, destination, overwrite):
        raise LogicError("C2F layer does not have a header file.")

    def generate_source_code(self, destination, overwrite):
        dest_path = Path(destination).resolve()
        if not dest_path.is_dir():
            self._error(f"{dest_path} does not exist!")
            raise RuntimeError("Directory does not exist")

        self._generate_c2f(dest_path, overwrite)
