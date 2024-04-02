import pathlib

from pathlib import Path
from pkg_resources import resource_filename

from . import AbcCodeGenerator
from . import LogicError
from . import TaskFunction
from .generate_packet_file import generate_packet_file
from .parse_helpers import parse_lbound_f
from . import (
    EXTERNAL_ARGUMENT, LBOUND_ARGUMENT, TILE_LBOUND_ARGUMENT,
    TILE_UBOUND_ARGUMENT, SCRATCH_ARGUMENT, F2C_TYPE_MAPPING,
    THREAD_INDEX_VAR_NAME, GRID_DATA_ARGUMENT, TILE_INTERIOR_ARGUMENT,
    TILE_ARRAY_BOUNDS_ARGUMENT, GRID_DATA_PTRS, SOURCE_DATATYPES,
    VECTOR_ARRAY_EQUIVALENT, TILE_ARGUMENTS_ALL, GRID_DATA_LBOUNDS,
)


# todo::
#   * update argument list order to use the tf argument list order.
class TaskFunctionCpp2CGenerator_cpu_F(AbcCodeGenerator):
    """
    Generates the Cpp2C layer for a TaskFunction using a CPU data item
    with a fortran based task function.
    """
    INDENT = 4 * " "
    C2F_ARG_LIST = "c2f_dummy_args"
    DUMMY_ARGS = "constructor_dummy_arguments"
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

        self.stree_opts = {
            'codePath': pathlib.Path.cwd(),
            'indentSpace': ' ' * indent,
            'verbose': False,
            'verbosePre': '/* ',
            'verbosePost': ' */',
        }

        self.connectors = {}
        self.tile_desc_name = "tileDesc"

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
            data_item_key = TaskFunction.DATA_ITEM_KEY
            class_header = \
                self._tf_spec.output_filenames[data_item_key]["header"]
            c2f_func = self._tf_spec.name + "_c2f"
            class_name = self._tf_spec.data_item_class_name
            cpp2c_func = self._tf_spec.name + "_cpp2c"

            outer.writelines([
                '/* _link:tf_cpp2c */\n'
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
        # convert to C type if necessary.
        dtype = F2C_TYPE_MAPPING.get(spec["type"], spec["type"])
        connectors[self.C2F_ARG_LIST].append(f"const {dtype} {arg}")
        connectors[self.REAL_ARGS].append(f"wrapper->{arg}_")

    def _fill_mdata_connectors(self, arg, spec, connectors: dict, saved):
        src = spec["source"]
        # # we combine tile_lo and tile_hi.
        if src == TILE_INTERIOR_ARGUMENT or src == TILE_ARRAY_BOUNDS_ARGUMENT:
            lo = 'tile_lo' if src == TILE_INTERIOR_ARGUMENT else "tile_loGC"
            hi = 'tile_hi' if src == TILE_INTERIOR_ARGUMENT else "tile_hiGC"
            connectors[self.C2F_ARG_LIST].append(f"const void* {src}")
            connectors[self.REAL_ARGS].append(f"static_cast<void*>({src})")
            lo_data = lo.replace("tile_", "") + "()"
            hi_data = hi.replace("tile_", "") + "()"
            combined = f"int {src}[] = {{\n{self.INDENT}"
            combined += f',\n{self.INDENT}'.join(
                '{0}->{1}.{3}(),{0}->{2}.{3}()'.format(
                    self.tile_desc_name, lo_data, hi_data, char
                )
                for char in ['I', 'J', 'K']
            ) + "\n}"

            self.connectors[self.CONSOLIDATE_TILE_DATA].append(combined)

            for bound in [lo, hi]:
                if bound not in saved:
                    funct = bound.replace("tile_", "")
                    self.connectors[self.TILE_DATA].append(
                        f"const auto {bound} = {self.tile_desc_name}->{funct}()"
                    )
                    saved.add(bound)
        else:
            alt_src = src
            if src == TILE_LBOUND_ARGUMENT:
                alt_src = "tile_loGC"
            elif src == TILE_LBOUND_ARGUMENT:
                alt_src = "tile_hiGC"
            tile_desc_func = alt_src.replace("tile_", '')
            tile_desc_name = self.tile_desc_name
            if alt_src not in saved:
                connectors[self.TILE_DATA].append(
                    f"const auto {arg} = {tile_desc_name}->{tile_desc_func}()"
                )
                saved.add(alt_src)

            # determines if we need to make a tile interior argument
            connectors[self.C2F_ARG_LIST].append(f"const void* {arg}_array")
            connectors[self.REAL_ARGS].append(
                f"static_cast<void*>({arg}_array)"
            )

            dtype = SOURCE_DATATYPES[src]
            if dtype in VECTOR_ARRAY_EQUIVALENT:
                raw = VECTOR_ARRAY_EQUIVALENT[dtype]
                connectors[self.CONSOLIDATE_TILE_DATA].append(
                    f"{raw} {arg}_array[] = {{\n{self.INDENT}{arg}.I(),\n"
                    f"{self.INDENT}{arg}.J(),\n"
                    f"{self.INDENT}{arg}.K()\n}}"
                )

    def _fill_lbound_connectors(self, arg, spec, connectors, saved):
        associated_var = spec["array"]
        var_spec = self._tf_spec.argument_specification(associated_var)
        # # we need to add loGC or hiGC to the list of tile data
        # set the size of array lbound based on source.
        # todo:: use starting index value
        lb = None
        words = None
        if var_spec["source"] == GRID_DATA_ARGUMENT:
            st_idx = var_spec["structure_index"][0].upper()
            gcells = self._tf_spec.n_guardcells
            lb, words = parse_lbound_f(GRID_DATA_LBOUNDS[st_idx].format(gcells))
        else:
            lb, words = parse_lbound_f(var_spec["lbound"])

        for word in words:
            alt = word
            if word == TILE_LBOUND_ARGUMENT or word == TILE_UBOUND_ARGUMENT:
                alt = \
                    "tile_loGC" if word == TILE_LBOUND_ARGUMENT else "tile_hiGC"
            if alt not in saved:
                tile_desc_func = alt.replace("tile_", "")
                tile_desc_name = self.tile_desc_name
                connectors[self.TILE_DATA].append(
                    f"const auto {word} = {tile_desc_name}->{tile_desc_func}()"
                )
                saved.add(word)

        lb = f"{{\n{self.INDENT}" + f',\n{self.INDENT}'.join(lb) + "\n}"
        lb = lb.replace(" ", "")
        connectors[self.CONSOLIDATE_TILE_DATA].append(
            f"int {arg}[] = {lb}"
        )
        connectors[self.C2F_ARG_LIST].append(f"const void* {arg}")
        connectors[self.REAL_ARGS].append(f"static_cast<void*>({arg})")

    def _fill_grid_connectors(self, arg, spec, connectors):
        dtype = SOURCE_DATATYPES[spec["source"]]
        grid_axis = spec["structure_index"][0]
        func = GRID_DATA_PTRS[grid_axis].format("tileDesc")
        connectors[self.C2F_ARG_LIST].append(f"const void* {arg}")
        connectors[self.TILE_DATA].append(f"{dtype}* {arg} = {func}")
        connectors[self.REAL_ARGS].append(f"static_cast<void*>({arg})")

    def _fill_scratch_connectors(self, arg, spec, connectors):
        dtype = F2C_TYPE_MAPPING.get(spec["type"], spec["type"])
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
        self.connectors[self.C2F_ARG_LIST] = []
        self.connectors[self.REAL_ARGS] = []
        self.connectors[self.TILE_DATA] = []
        self.connectors[self.CONSOLIDATE_TILE_DATA] = []
        self.connectors[self.AC_SCRATCH] = []
        # save names of any tile metadata that was already added.
        saved = set()
        for var in self._tf_spec.dummy_arguments:
            spec = self._tf_spec.argument_specification(var)
            src = spec["source"]
            if src == EXTERNAL_ARGUMENT:
                self._fill_external_connectors(var, spec, self.connectors)
            elif src in TILE_ARGUMENTS_ALL:
                self._fill_mdata_connectors(var, spec, self.connectors, saved)
            elif src == LBOUND_ARGUMENT:
                self._fill_lbound_connectors(var, spec, self.connectors, saved)
            elif src == GRID_DATA_ARGUMENT:
                self._fill_grid_connectors(var, spec, self.connectors)
            elif src == SCRATCH_ARGUMENT:
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
            output, [outer, self.tf_cpp2c_template, helper],
            overwrite, self._logger
        )