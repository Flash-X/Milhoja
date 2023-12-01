import pathlib

from pathlib import Path
from pkg_resources import resource_filename

from . import AbcCodeGenerator
from . import LogicError
from . import TaskFunction
from .generate_packet_file import generate_packet_file
from .parse_helpers import parse_lbound_f
from . import GRID_DATA_FUNC_MAPPING
from . import SOURCE_DATATYPE_MAPPING
from . import (
    TILE_LO_ARGUMENT,
    TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT,
    TILE_UBOUND_ARGUMENT,
    F_HOST_EQUIVALENT,
    THREAD_INDEX_VAR_NAME
)


class TaskFunctionCpp2CGenerator_cpu_f(AbcCodeGenerator):
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
        self.tf_cpp2c_name = 'cg-tpl.tf_cpp2c.cxx'
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
            f"cg-tpl.{self._tf_spec.data_item_class_name}_outer.cpp"
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
                f'/* _connector:tf_cpp2c */\n'
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

    def _generate_helper_template(self, destination: Path, overwrite) -> Path:
        helper_template = destination.joinpath(
            f"cg-tpl.{self._tf_spec.data_item_class_name}_helper.cpp"
        ).resolve()

        if helper_template.is_file():
            self._warn(f"{str(helper_template)} already exists.")
            if overwrite:
                self._error("Overwrite is set to False.")
                raise FileExistsError()

        # generate all connectors
        # append to c2f arg list.
        self.connectors[self.C2F_ARG_LIST] = []
        self.connectors[self.REAL_ARGS] = []
        for ext in self._tf_spec.external_arguments:
            spec = self._tf_spec.argument_specification(ext)
            dtype = spec["type"]
            if dtype == "real":
                dtype = "milhoja::Real"
            self.connectors[self.C2F_ARG_LIST].append(f"const {dtype} {ext}")
            self.connectors[self.REAL_ARGS].append(f"wrapper->{ext}")

        # generate tile_data connector.
        tile_desc_name = "tileDesc"
        metadata = self._tf_spec.tile_metadata_arguments
        metadata_source_mapping = {}
        interior = [None, None]

        self.connectors[self.TILE_DATA] = []
        for mdata in self._tf_spec.tile_metadata_arguments:
            arg_spec = self._tf_spec.argument_specification(mdata)
            source = arg_spec["source"]
            tile_desc_func = source

            # todo:: temporary
            if source == TILE_LBOUND_ARGUMENT:
                tile_desc_func = "tile_loGC"
            elif source == TILE_UBOUND_ARGUMENT:
                tile_desc_func = "tile_hiGC"

            # determines if we need to make a tile interior argument
            # todo: Not sure what happens if lo or hi appears twice.
            #       TaskFunction handles this beforehand?
            if source == TILE_LO_ARGUMENT:
                interior[0] = mdata
            elif source == TILE_HI_ARGUMENT:
                interior[1] = mdata

            self.connectors["tile_data"].append(
                f"const milhoja::{SOURCE_DATATYPE_MAPPING[source]} {mdata} = "
                f"{tile_desc_name}->{tile_desc_func.replace('tile_', '')}()"
            )

            dtype = SOURCE_DATATYPE_MAPPING[source]
            if dtype in F_HOST_EQUIVALENT:
                raw = F_HOST_EQUIVALENT[dtype]
                if raw == "real":
                    raw = "milhoja::Real"
                self.connectors[self.TILE_DATA].append(
                    f"{raw} {source}_array[] = {{{mdata}.I(),{mdata}.J()," 
                    f"{mdata}.K()}}"
                )
                self.connectors[self.REAL_ARGS].append(
                    f"static_cast<void*>({source}_array)"
                )

        # both lo and hi are in metadata.
        if len(interior) == 2:
            combined = "int tile_interior[] = {"
            combined += ','.join(
                f'{interior[0]}.{char}(), {interior[1]}.{char}()'
                for char in ['I', 'J', 'K']
            ) + "}"
            self.connectors[self.TILE_DATA].append(combined)

        # generate acquire_scratch connector and update other connectors.
        # todo: a small optimization would be to check if the same lbound
        #       is used in other arrays and to only create 1 bound array.
        self.connectors["acquire_scratch"] = []
        class_name = self._tf_spec.data_item_class_name
        for scratch in self._tf_spec.scratch_arguments:
            spec = self._tf_spec.argument_specification(scratch)
            lbound = parse_lbound_f(spec["lbound"])
            dtype = spec["type"]
            dtype = dtype.capitalize() if dtype == "real" else dtype
            dtype = f"milhoja::{dtype}"

            self.connectors["acquire_scratch"].append(
                f"{dtype}* {scratch} = static_cast<{dtype}*>("
                f"{class_name}::{scratch}) + {class_name}::{scratch.upper()}"
                f"_SIZE_ * {THREAD_INDEX_VAR_NAME}"
            )
            lb_name = f'lb_{scratch}'
            self.connectors[self.TILE_DATA].append(
                f'int {lb_name}[] = {{'
                f"{','.join(lbound)}"
                '}'
            )
            self.connectors[self.REAL_ARGS].extend([
                f'static_cast<void*>({lb_name})',
                f'static_cast<void*>({scratch})'
            ])
            self.connectors[self.C2F_ARG_LIST].extend([
                f'const void* {lb_name}',
                f'const void* {scratch}'
            ])

        # generate helper template
        with open(helper_template, 'w') as helper:
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
