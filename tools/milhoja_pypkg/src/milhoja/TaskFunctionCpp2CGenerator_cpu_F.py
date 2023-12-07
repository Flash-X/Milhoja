import pathlib

from pathlib import Path
from pkg_resources import resource_filename
from copy import deepcopy
from collections import OrderedDict

from . import AbcCodeGenerator
from . import LogicError
from . import TaskFunction
from .generate_packet_file import generate_packet_file
from .parse_helpers import parse_lbound_f
from . import GRID_DATA_FUNC_MAPPING
from . import SOURCE_DATATYPE_MAPPING
from . import (
    EXTERNAL_ARGUMENT,
    TILE_LO_ARGUMENT,
    TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT,
    TILE_UBOUND_ARGUMENT,
    SCRATCH_ARGUMENT,
    F_HOST_EQUIVALENT,
    THREAD_INDEX_VAR_NAME,
    GRID_DATA_ARGUMENT,
    TILE_INTERIOR_ARGUMENT,
    TILE_ARRAY_BOUNDS_ARGUMENT
)


# todo::
#   * update argument list order to use the tf argument list order.
class TaskFunctionCpp2CGenerator_cpu_F(AbcCodeGenerator):
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
        source = tf_spec.output_filenames[TaskFunction.CPP_TF_KEY]["source"]
        self.tf_cpp2c_name = source
        self.tf_cpp2c_template = Path(
            resource_filename(
                __package__, 'templates/cg-tpl.tf_cpp2c.cpp'
            )
        ).resolve()

        self.stree_opts  = {
            'codePath': pathlib.Path.cwd(),
            'indentSpace': ' ' * indent,
            'verbose': False,
            'verbosePre': '/* ',
            'verbosePost': ' */',
        }

        self.connectors = {}

        super().__init__(
            tf_spec, header, source, indent,
            "Milhoja TF Cpp2C", logger
        )

    def _generate_outer_template(self, destination: Path, overwrite) -> Path:
        """
        Generates the outer template for the cpp2c layer and returns the path
        containing the template file.

        :param destination: The destination folder for the template.
        :param overwrite: Flag for overwriting the template file if it exists
        """
        outer_template = destination.joinpath(
            f"cg-tpl.{self._tf_spec.name}_outer.cpp"
        ).resolve()

        if outer_template.is_file():
            self._warn(f"{str(outer_template)} already exists.")
            if overwrite:
                self._error("Overwrite is set to False.")
                raise FileExistsError()

        # generate outer template
        with open(outer_template, 'w') as outer:
            class_header = self._tf_spec.output_filenames\
                [TaskFunction.DATA_ITEM_KEY]["header"]
            c2f_func = self._tf_spec.name + "_c2f"
            class_name = self._tf_spec.data_item_class_name
            cpp2c_func = self._tf_spec.name + "_cpp2c"

            outer.writelines([
                f'/* _link:tf_cpp2c */\n'
                '/* _param:data_item_header_file_name = ',
                f'{class_header} */\n',
                '/* _param:c2f_function_name = ',
                f'{c2f_func} */\n'
                '/* _param:cpp2c_function_name = ',
                f'{cpp2c_func} */\n'
                '/* _param:data_item_class = ',
                f'{class_name} */\n'
            ])

        return outer_template

    def _fill_external_connectors(self, arg, spec, connectors: dict):
        dtype = spec["type"]
        if dtype == "real":
            dtype = "milhoja::Real"
        connectors[self.C2F_ARG_LIST].append(f"const {dtype} {arg}")
        connectors[self.REAL_ARGS].append(f"wrapper->{arg}_")

    def _fill_mdata_connectors(self, arg, spec, connectors: dict, combine_arrays, saved=set()):
        tile_desc_name = "tileDesc"
        # todo:: does not account for other tile metadata.
        source = spec["source"]
        associated_var = spec.get("array", None)

        # if this is an lbound for a different array
        if associated_var:
            var_spec = self._tf_spec.argument_specification(associated_var)
            # # we need to add loGC or hiGC to the list of tile data
            # set the size of array lbound based on source.
            # todo:: use starting index value
            if var_spec["source"] == GRID_DATA_ARGUMENT:
                if arg not in saved:
                    self.connectors[self.TILE_DATA].append(
                        f"const milhoja::IntVect tile_lbound = "
                        f"{tile_desc_name}->loGC()"
                    )
                lb = parse_lbound_f("(tile_lbound, 1)")
            else:
                lb = parse_lbound_f(var_spec["lbound"])
            connectors[self.CONSOLIDATE_TILE_DATA].append(
                f"int {arg}[] = {{{','.join(lb)}}}"
            )
            connectors[self.C2F_ARG_LIST].append(f"const void* {arg}")
            connectors[self.REAL_ARGS].append(f"static_cast<void*>({arg})")
            saved.add(arg)
        # # otherwise it's a normal tile metadata
        else:
            saved.add(source)
            alt_src = source
            if source == TILE_LBOUND_ARGUMENT:
                alt_source = "tile_loGC"
            elif source == TILE_LBOUND_ARGUMENT:
                alt_source = "tile_hiGC"

            tile_desc_func = source.replace("tile_", '')
            connectors[self.TILE_DATA].append(
                f"const milhoja::{SOURCE_DATATYPE_MAPPING[source]} {arg} = "
                f"{tile_desc_name}->{tile_desc_func}()"
            )

            # # we combine tile_lo and tile_hi.
            if combine_arrays[0] and \
            (source == TILE_LO_ARGUMENT or source == TILE_HI_ARGUMENT):
                if TILE_INTERIOR_ARGUMENT not in saved:
                    name = TILE_INTERIOR_ARGUMENT
                    connectors[self.C2F_ARG_LIST].append(f"const void* {name}")
                    connectors[self.REAL_ARGS].append(f"static_cast<void*>({name})")
                    combined = f"int {name}[] = {{"
                    combined += ','.join(
                        f'tile_lo.{char}(),tile_hi.{char}()'
                        for char in ['I', 'J', 'K']
                    ) + "}"
                    self.connectors[self.CONSOLIDATE_TILE_DATA].append(combined)
                    saved.add(TILE_INTERIOR_ARGUMENT)
            elif combine_arrays[1] and \
            source == TILE_LBOUND_ARGUMENT or source == TILE_UBOUND_ARGUMENT:
                if TILE_ARRAY_BOUNDS_ARGUMENT not in saved:
                    name = TILE_ARRAY_BOUNDS_ARGUMENT
                    connectors[self.C2F_ARG_LIST].append(f"const void* {name}")
                    connectors[self.REAL_ARGS].append(f"static_cast<void*>({name})")
                    combined = f"int {name}[] = {{"
                    combined += ','.join(
                        f'tile_lbound.{char}(),tile_ubound.{char}()'
                        for char in ['I', 'J', 'K']
                    ) + "}"
                    connectors[self.CONSOLIDATE_TILE_DATA].append(combined)
                    saved.add(TILE_ARRAY_BOUNDS_ARGUMENT)
            else:
                # determines if we need to make a tile interior argument
                # todo: Not sure what happens if lo or hi appears twice.
                #       TaskFunction handles this beforehand?
                connectors[self.C2F_ARG_LIST].append(f"const void* {arg}_array")
                connectors[self.REAL_ARGS].append(
                    f"static_cast<void*>({arg}_array)"
                )

                dtype = SOURCE_DATATYPE_MAPPING[source]
                if dtype in F_HOST_EQUIVALENT:
                    raw = F_HOST_EQUIVALENT[dtype]
                    if raw == "real":
                        raw = "milhoja::Real"
                    connectors[self.CONSOLIDATE_TILE_DATA].append(
                        f"{raw} {arg}_array[] = {{{arg}.I(),{arg}.J()," 
                        f"{arg}.K()}}"
                    )
                saved.add(arg)

    def _fill_grid_connectors(self, arg, spec, connectors):
        dtype = "milhoja::Real"
        grid_axis = spec["structure_index"][0]
        func = GRID_DATA_FUNC_MAPPING[grid_axis].format("tileDesc")
        connectors[self.C2F_ARG_LIST].append(f"const void* {arg}")
        connectors[self.TILE_DATA].append(f"{dtype}* {arg} = {func}")
        connectors[self.REAL_ARGS].append(f"static_cast<void*>({arg})")

    def _fill_scratch_connectors(self, arg, spec, connectors):
        dtype = spec["type"]
        dtype = dtype.capitalize() if dtype == "real" else dtype
        dtype = f"{dtype}"
        class_name = self._tf_spec.data_item_class_name

        connectors["acquire_scratch"].append(
            f"{dtype}* {arg} = static_cast<{dtype}*>("
            f"{class_name}::{arg.lower()}_) + {class_name}::{arg.upper()}"
            f"_SIZE_ * {THREAD_INDEX_VAR_NAME}"
        )
        connectors[self.REAL_ARGS].append(f'static_cast<void*>({arg})')
        connectors[self.C2F_ARG_LIST].append(f'const void* {arg}')

    def _generate_helper_template(self, destination: Path, overwrite) -> Path:
        helper_template = destination.joinpath(
            f"cg-tpl.{self._tf_spec.name}_helper.cpp"
        ).resolve()

        if helper_template.is_file():
            self._warn(f"{str(helper_template)} already exists.")
            if overwrite:
                self._error("Overwrite is set to False.")
                raise FileExistsError()

        # generate tile_data connector.
        combine_arrays = self._tf_spec.use_combined_array_bounds
        self.connectors[self.C2F_ARG_LIST] = []
        self.connectors[self.REAL_ARGS] = []
        self.connectors[self.TILE_DATA] = []
        self.connectors[self.CONSOLIDATE_TILE_DATA] = []
        self.connectors[self.AC_SCRATCH] = []
        for var in self._tf_spec.dummy_arguments:
            spec = self._tf_spec.argument_specification(var)
            source = spec["source"]
            if source == EXTERNAL_ARGUMENT:
                self._fill_external_connectors(var, spec, self.connectors)
            elif source.startswith("tile_"):
                self._fill_mdata_connectors(var, spec, self.connectors, combine_arrays)
            elif source == GRID_DATA_ARGUMENT:
                self._fill_grid_connectors(var, spec, self.connectors)
            elif source == SCRATCH_ARGUMENT:
                self._fill_scratch_connectors(var, spec, self.connectors)

        # generate helper template
        with open(helper_template, 'w') as helper:
            # dummy arg list
            helper.write(f"/* _connector:{self.C2F_ARG_LIST} */\n")
            arg_list = self.connectors[self.C2F_ARG_LIST]
            helper.write(', \n'.join(arg_list))
            helper.write("\n")
            del self.connectors[self.C2F_ARG_LIST]

            # passed in args
            helper.write(f"/* _connectors:{self.REAL_ARGS} */\n")
            real_args = self.connectors[self.REAL_ARGS]
            helper.write(', \n'.join(real_args))
            helper.write("\n")
            del self.connectors[self.REAL_ARGS]

            for section, code in self.connectors.items():
                helper.write(f"/* _connector:{section} */\n")
                helper.writelines([line + ";\n" for line in code])
                helper.write("\n")

        return helper_template

    def generate_header_code(self, destination, overwrite):
        raise LogicError("Cpp2C TF does not have a header file.")

    def generate_source_code(self, destination, overwrite):
        dest_path = Path(destination).resolve()
        if not dest_path.is_dir():
            self._error(f"{dest_path} does not exist!")
            raise RuntimeError("Directory does not exist")

        outer = self._generate_outer_template(dest_path, overwrite)
        helper = self._generate_helper_template(dest_path, overwrite)
        output = dest_path.joinpath(self.tf_cpp2c_name)

        generate_packet_file(
            output, self.stree_opts, [outer, self.tf_cpp2c_template, helper],
            overwrite, self._logger
        )
