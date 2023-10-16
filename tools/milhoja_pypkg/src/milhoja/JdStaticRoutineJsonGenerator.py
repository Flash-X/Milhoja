from .StaticRoutineJsonGenerator import StaticRoutineJsonGenerator

class JdStaticRoutineJsonGenerator(StaticRoutineJsonGenerator):
    """
    Generates a json for a StaticRoutine using json directives.
    """
    def __init__(self, destination: str, files_to_parse: list, combine: bool, log_level):
        super().__init__(destination, files_to_parse, combine, log_level)

    def generate_routine_json(self, interface_name, routine_file) -> dict:
        base = {}
        return base