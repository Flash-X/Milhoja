from pathlib import Path

from . import AbcCodeGenerator
from . import LOG_LEVEL_BASIC_DEBUG
from . import INTERNAL_ARGUMENT
from . import TaskFunction
from . import LogicError


class DataPacketC2FModuleGenerator(AbcCodeGenerator):

    _TYPE_MAPPING = {
        "real": "real(MILHOJA_REAL)",
        "int": "integer(MILHOJA_INT)",
        "bool": "logical"
    }

    # since this is specifically for the data packet generator it
    # should be fine to pull specifically from the data packet
    # external arguments. The Cpp2C and C2F layers ultimately should not
    # do that, however, since they're more task function related.
    def __init__(self, tf_spec, indent, logger, external_args):
        if tf_spec.language.lower() == "c++":
            raise LogicError("No mod file for C++.")

        file_name = tf_spec.output_filenames[
            TaskFunction.DATA_ITEM_KEY
        ]["module"]
        super().__init__(
            tf_spec, "", file_name,
            indent, "Milhoja DataPacket C2F Module",
            logger
        )
        self.INDENT = " " * indent
        self._externals = external_args

    def generate_header_code(self, destination, overwrite):
        raise LogicError(
            "generate_header_code not implemented for module generator."
        )

    def generate_source_code(self, destination, overwrite):
        destination_path = Path(destination).resolve()
        if not destination_path.is_dir():
            raise FileNotFoundError(
                f"{destination_path} does not exist."
            )

        mod_path = destination_path.joinpath(self.source_filename).resolve()
        if mod_path.is_file():
            self._log(f"{mod_path} already exists.", LOG_LEVEL_BASIC_DEBUG)
            if not overwrite:
                raise FileExistsError("Overwrite is set to False.")

        with open(mod_path, 'w') as module:
            dataitem_name = self._tf_spec.data_item_class_name
            module_name = f"{dataitem_name}_c2f_mod"
            tf_name = self._tf_spec.name

            # todo:: get these from tf_spec.
            instance = f"instantiate_{tf_name}_packet_c"
            delete = f"delete_{tf_name}_packet_c"
            release = f"release_{tf_name}_extra_queue_c"

            # declare interface functions for time advance unit
            module.write(f"module {module_name}\n")
            module.write(f"{self.INDENT}implicit none\n")
            module.write(f"{self.INDENT}private\n\n")
            module.write(f"{self.INDENT}public :: {instance}\n")
            module.write(f"{self.INDENT}public :: {delete}\n")
            module.write(f"{self.INDENT}public :: {release}\n\n")

            # write functions
            module.write(f"{self.INDENT}interface\n")
            module.write(f"{self.INDENT * 2}function {instance}( &\n")

            arg_list = []
            var_declarations = []
            for var, data in self._externals.items():
                if data["source"] == INTERNAL_ARGUMENT:
                    continue
                dtype = data["type"]
                name = f"C_{var}"
                arg_list.append(name)
                var_declarations.append(
                    f"{self._TYPE_MAPPING[dtype]}, "
                    f"intent(IN), value :: {name}"
                )

            args = f' &\n{self.INDENT * 3}'.join(arg_list)
            module.write(f'{self.INDENT * 3}' + args)
            module.write(
                f", &\n{self.INDENT*3}C_packet &\n"
                f"{self.INDENT * 2})result(C_ierr) bind (c)\n"
            )
            module.write(f"{self.INDENT * 3}use iso_c_binding, ONLY: C_PTR\n")
            module.write(
                f"{self.INDENT * 3}use milhoja_types_mod, "
                "ONLY: MILHOJA_INT, MILHOJA_REAL\n"
            )
            vars = f'\n{self.INDENT * 3}'.join(var_declarations)
            module.write(f"{self.INDENT * 3}" + vars + "\n")
            module.write(
                f"{self.INDENT * 3}type(C_PTR), intent(IN) :: C_packet\n"
            )
            module.write(f"{self.INDENT * 3}integer(MILHOJA_INT) :: C_ierr\n")
            module.write(f"{self.INDENT * 2}end function {instance}\n\n")

            module.write(f"{self.INDENT * 2}function {delete}(")
            module.write("C_packet) result(C_ierr) bind (c)\n")

            module.writelines([
                f"{self.INDENT * 3}use iso_c_binding, ONLY : C_PTR\n",
                f"{self.INDENT * 3}"
                "use milhoja_types_mod, ONLY : MILHOJA_INT\n",
                f"{self.INDENT * 3}"
                "type(C_PTR), intent(IN), value :: C_packet\n"
                f"{self.INDENT * 3}integer(MILHOJA_INT) :: C_ierr\n"
            ])

            module.write(f"{self.INDENT * 2}end function {delete}\n")
            module.write(f"{self.INDENT}end interface\n\n")
            # end

            # write interface release function for task function
            module.write(f"{self.INDENT}interface\n")
            module.write(f"{self.INDENT * 2}function {release}(")
            module.write("C_packet, C_id) result(C_ierr) bind(c)\n")

            # write function
            module.writelines([
                f"{self.INDENT * 3}use iso_c_binding, ONLY : C_PTR\n",
                f"{self.INDENT * 3}"
                "use milhoja_types_mod, ONLY : MILHOJA_INT\n",
                f"{self.INDENT * 3}"
                "type(C_PTR), intent(IN), value :: C_packet\n"
                f"{self.INDENT * 3}"
                "integer(MILHOJA_INT), intent(IN), value :: C_id\n"
                f"{self.INDENT * 3}integer(MILHOJA_INT) :: C_ierr\n"
            ])

            module.write(f"{self.INDENT * 2}end function {release}\n")
            module.write(f"{self.INDENT}end interface\n\n")
            module.write(f"end module {module_name}")
