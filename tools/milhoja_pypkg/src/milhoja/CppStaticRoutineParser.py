import json

from .StaticRoutineParser import StaticRoutineParser
from . import LOG_LEVEL_BASIC_DEBUG

class CppStaticRoutineParser(StaticRoutineParser):
    def __init__(self, destination: str, files_to_parse: list, log_level):
        """
        Constructor. 

        :param str destination: The desination folder of the json outputs.
        :param list files_to_parse: The static fortran routine paths to parse.
        """ 
        super().__init__(destination, files_to_parse, log_level)
        
    def __parse_from_directives(self, routine_file) -> dict:
        variables = {}
        json_string = ""
        for line in routine_file:
            if "$flashx" in line:
                json_string += line[line.find("$flashx")+len("$flashx"):].strip()
        json_string = "{" + json_string + "}"
        self._logger.log(self._TOOL_NAME, json_string, LOG_LEVEL_BASIC_DEBUG)
        variables = json.loads(json_string)
        return variables

    def parse_routine(self, routine_file) -> dict:
        return self.__parse_from_directives(routine_file)
        read_vars = set()
        write_vars = set()

        for line in routine_file.read():
            ...

        for item in read_vars:
            status = "R"
            if item in write_vars:
                status = "RW"

        status = "W"
        for item in write_vars:
            if item in read_vars:
                continue

        return {}
            

        