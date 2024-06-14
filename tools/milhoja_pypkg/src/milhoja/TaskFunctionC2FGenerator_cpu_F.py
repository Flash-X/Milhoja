import pathlib

from dataclasses import dataclass
from pathlib import Path
from . import AbcCodeGenerator
from . import LogicError
from . import TaskFunction
from .parse_helpers import (
    parse_lbound_f, parse_extents, get_array_size, get_initial_index
)
from . import (
    EXTERNAL_ARGUMENT, LBOUND_ARGUMENT, GRID_DATA_ARGUMENT,
    SCRATCH_ARGUMENT, TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT,
    TILE_ARGUMENTS_ALL, VECTOR_ARRAY_EQUIVALENT, SOURCE_DATATYPES,
    GRID_DATA_EXTENTS, GRID_DATA_LBOUNDS, C2F_TYPE_MAPPING
)


# todo::
#   * There is also a dataclass in the DataPacket version of the C2F
#     generator. Maybe it would be worth it to see we can combine these two?
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

        self.stree_opts = {
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
        """
        Builds and returns a ConversionData object for an external argument,
        for use in the C2F layer.

        :param arg: The argument name.
        :param arg_spec: The argument specification. 'external' is the assumed
                         source.
        """
        dtype = arg_spec["type"]
        if dtype == "milhoja::Real":
            dtype = "real"
        shape = \
            parse_extents(arg_spec["extents"]) if arg_spec["extents"] else []
        return ConversionData(
            cname=f"C_{arg}",
            fname=f"F_{arg}",
            dtype=dtype,
            is_pointer=True if shape else False,
            shape=shape
        )

    def _get_tmdata_info(self, arg, arg_spec) -> ConversionData:
        """
        Builds and returns a ConversionData object for a tile metadata
        argument, for use in the C2F layer.

        :param arg: The argument name.
        :param arg_spec: The argument specification. Any tile metadata source
                         is the assumed source.
        """
        src = arg_spec["source"]
        name_key = arg
        dtype = VECTOR_ARRAY_EQUIVALENT.get(
            SOURCE_DATATYPES[src], SOURCE_DATATYPES[src]
        )
        dtype = C2F_TYPE_MAPPING[dtype]
        shape = ["MILHOJA_MDIM"]
        if src == TILE_INTERIOR_ARGUMENT or src == TILE_ARRAY_BOUNDS_ARGUMENT:
            name_key = src
            shape.insert(0, "2")

        return ConversionData(
            cname=f"C_{name_key}",
            fname=f"F_{name_key}",
            dtype=dtype,
            is_pointer=True,
            shape=shape
        )

    def _get_lbound_info(self, arg, arg_spec) -> ConversionData:
        """
        Builds and returns a ConversionData object for an lbound argument,
        for use in the C2F layer.

        :param arg: The argument name.
        :param arg_spec: The argument specification. Any lbound source
                         is the assumed source.
        """
        arr = arg_spec["array"]
        array_arg_spec = self._tf_spec.argument_specification(arr)
        arr_src = array_arg_spec["source"]
        lbound = ""

        if arr_src == GRID_DATA_ARGUMENT:
            st_idx = array_arg_spec["structure_index"][0].upper()
            vars_in = array_arg_spec.get('variables_in', None)
            vars_out = array_arg_spec.get('variables_out', None)
            init = get_initial_index(vars_in, vars_out)
            lbound = GRID_DATA_LBOUNDS[st_idx].format(init)
        else:
            lbound = array_arg_spec[LBOUND_ARGUMENT]

        lbound, _ = parse_lbound_f(lbound)
        lbound = [item.replace("tile_", "") for item in lbound]
        bound_size = len(lbound)
        return ConversionData(
            cname=f"C_{arg}",
            fname=f"F_{arg}",
            dtype="integer",
            is_pointer=True,
            shape=[str(bound_size)]
        )

    def _get_grid_info(self, arg, arg_spec) -> ConversionData:
        """
        Builds and returns a ConversionData object for a grid data
        argument, for use in the C2F layer.

        :param arg: The argument name.
        :param arg_spec: The argument specification. 'grid_data' is the
                         assumed source.
        """
        dtype = SOURCE_DATATYPES[arg_spec["source"]]
        mask_in = arg_spec.get("variables_in", [])
        mask_out = arg_spec.get("variables_out", [])
        size = get_array_size(mask_in, mask_out)
        st_index = arg_spec["structure_index"][0].upper()
        block_size = self._tf_spec.block_interior_shape
        gcells = self._tf_spec.n_guardcells
        grid_exts = GRID_DATA_EXTENTS

        shape = [
            * [
                ext.format(block_size[i], gcells)
                for i, ext in enumerate(grid_exts[st_index])
            ],
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
        """
        Builds and returns a ConversionData object for a scratch
        argument, for use in the C2F layer.

        :param arg: The argument name.
        :param arg_spec: The argument specification. 'scratch' is the assumed
                         source.
        """
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
        """
        Generates the C to Fortran layer for a task function.

        :arg destination: The location to save the file.
        :arg overwrite: If set, overwrite anything at the source destination.
        """
        c2f_file = destination.joinpath(self.source_filename).resolve()

        if c2f_file.is_file():
            self._warn(f"{str(c2f_file)} already exists.")
            if not overwrite:
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
                src = arg_spec["source"]
                if src == EXTERNAL_ARGUMENT:
                    conversion_info = self._get_external_info(arg, arg_spec)
                elif src in TILE_ARGUMENTS_ALL:
                    conversion_info = self._get_tmdata_info(arg, arg_spec)
                elif src == LBOUND_ARGUMENT:
                    conversion_info = self._get_lbound_info(arg, arg_spec)
                elif src == GRID_DATA_ARGUMENT:
                    conversion_info = self._get_grid_info(arg, arg_spec)
                elif src == SCRATCH_ARGUMENT:
                    conversion_info = self._get_scratch_info(arg, arg_spec)
                else:
                    raise NotImplementedError(f"{src} not implemented.")

                if conversion_info:
                    arg_conversion_info.append(conversion_info)

            routine_name = self._tf_spec.c2f_layer_name
            c2f.write('#include "Milhoja.h"\n\n')
            c2f.write(f'subroutine {routine_name}( &\n')
            arg_list = f', &\n{self.INDENT}'.join(func_args)
            arg_list = f', &\n{self.INDENT}'.join([
                info.cname for info in arg_conversion_info
            ])
            c2f.write(
                f'{self.INDENT}{arg_list} &\n)'
                f'bind(c, name="{routine_name}")\n'
            )

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
                    # need to call conversion function,
                    # so we need to do this or use a dtype -> cast func
                    # map.
                    dtype = info.dtype if info.dtype != "integer" else "int"
                    set_values.append(
                        f"{self.INDENT}{info.fname} = "
                        f"{dtype.upper()}({info.cname})"
                    )
                else:
                    set_values.append(
                        f"{self.INDENT}CALL C_F_POINTER({info.cname}, "
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
        """No header code for C2F layer, raises LogicError."""
        raise LogicError("C2F layer does not have a header file.")

    def generate_source_code(self, destination, overwrite):
        """
        Wrapper around the actual code generation function for generating
        source code.
        """
        dest_path = Path(destination).resolve()
        if not dest_path.is_dir():
            self._error(f"{dest_path} does not exist!")
            raise RuntimeError("Directory does not exist")

        self._generate_c2f(dest_path, overwrite)
