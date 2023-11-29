from pathlib import Path

from . import AbcCodeGenerator
from . import TaskFunction


class DataPacketC2FModuleGenerator(AbcCodeGenerator):

    def __init__(self, tf_spec, indent, logger, external_args):
        file_name = tf_spec.data_item_class_name + "_C2F_mod.F90"
        super().__init__(
            # todo:: use name from tf_spec
            tf_spec, "", file_name,
            indent, "Milhoja C2F Module",
            logger
        )
        self.INDENT = " " * indent
        self._externals = external_args

    def generate_header_code(self, destination, overwrite):
        raise NotImplementedError(
            "generate_header_code not implemented for module generator."
        )

    def generate_source_code(self, destination, overwrite):

        # destination_path = Path(destination).resolve()
        # if not destination_path.is_dir():
        #     raise RuntimeError(
        #         f"{destination_path} does not exist."
        #     )
        # c2f_path = destination_path.joinpath(self.c2f_file).resolve()

        with open(destination, 'w') as module:
            dataitem_name = self._tf_spec.data_item_class_name
            module_name = f"{dataitem_name}_C2F_mod"
            tf_name = self._tf_spec.name

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
            module.write(f"{self.INDENT}interface")
            module.write(f"{self.INDENT * 2}function {instance}(\n")
            # todo:: write arg list
            module.write(") result(C_ierr) bind (c)\n")
            module.write(f"{self.INDENT * 3}\n")
            # todo:: write instance
            module.write(f"{self.INDENT * 2}end function {instance}\n\n")

            module.write(f"{self.INDENT * 2}function {delete}(\n")
            module.write("C_packet) result(C_ierr) bind (c)\n")

            module.writelines([
                f"{self.INDENT * 3}use iso_c_binding, ONLY : C_PTR",
                f"{self.INDENT * 3}use milhoja_types_mod, ONLY : MILHOJA_INT",
                f"{self.INDENT * 3}type(C_PTR), intent(IN), value :: C_packet\n"
                f"{self.INDENT * 3}integer(MILHOJA_INT) :: C_ierr\n"
            ])

            module.write(f"{self.INDENT * 2}end function {delete}\n")
            module.write(f"{self.INDENT}end interface\n\n")
            # end

            # write interface release function for task function
            module.write(f"{self.INDENT}interface\n")
            module.write(f"{self.INDENT * 2}function {release}(\n")
            module.write(f"C_packet, C_id) result(C_ierr) bind(c)\n")

            # write function
            module.writelines([
                f"{self.INDENT * 3}use iso_c_binding, ONLY : C_PTR",
                f"{self.INDENT * 3}use milhoja_types_mod, ONLY : MILHOJA_INT",
                f"{self.INDENT * 3}type(C_PTR), intent(IN), value :: C_packet\n"
                f"{self.INDENT * 3}integer(MILHOJA_INT), intent(IN), value :: C_id"
                f"{self.INDENT * 3}integer(MILHOJA_INT) :: C_ierr\n"
            ])

            module.write(f"{self.INDENT * 2}end function {release}\n")
            module.write(f"{self.INDENT}end interface\n\n")
            module.write(f"end module {module_name}")