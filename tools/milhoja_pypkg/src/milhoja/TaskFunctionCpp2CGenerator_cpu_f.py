import pathlib

from pathlib import Path
from pkg_resources import resource_filename

from . import AbcCodeGenerator
from . import LogicError
from . import TaskFunction
from . import generate_packet_file


class TaskFunctionCpp2CGenerator_cpu_f(AbcCodeGenerator):
    """
    Generates the Cpp2C layer for a TaskFunction using a CPU data item
    with a fortran based task function.
    """
    def __init__(self, tf_spec: TaskFunction, indent, logger):
        header = None
        source = tf_spec.output_filenames[TaskFunction.CPP_TF_KEY]["source"]
        self.tf_cpp2c_name = 'cg-tpl.tf_cpp2c.cxx'
        self.tf_cpp2c_template = Path(
            resource_filename(
                __package__, 'templates/cg-tpl.tf_cpp2c.cxx'
            )
        ).resolve()

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
            f"cg-tpl.{self._tf_spec.data_item_class_name}_outer.cpp"
        ).resolve()

        if helper_template.is_file():
            self._warn(f"{str(helper_template)} already exists.")
            if overwrite:
                self._error("Overwrite is set to False.")
                raise FileExistsError()

        # generate helper template
        with open(helper_template, 'w') as helper:
            ...

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
