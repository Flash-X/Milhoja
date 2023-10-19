import json
import re

from .StaticRoutineParser import StaticRoutineParser
from . import LOG_LEVEL_BASIC_DEBUG

# TODO: In order to get a list of unknown variables,
#       we need the list of all grid data structures being used,
#       and the list of all variable names. (VELX_VAR, VELY_VAR, etc.)

class FortranStaticRoutineParser(StaticRoutineParser):
    __FIND_RW_EXPR_START = r"(?<![^\s(_-])"
    __FIND_RW_EXPR_END = r"(?![^\s(_-])([^)]*)"

    def __init__(self, destination: str, files_to_parse: list, log_level, delimiter: str,
                 grid_vars: set, unks: set):
        """
        Constructor. 

        :param str destination: The desination folder of the json outputs.
        :param list files_to_parse: The static fortran routine paths to parse.
        """ 
        super().__init__(destination, files_to_parse, log_level)
        self.__DELIMITER = delimiter
        self.__GRID_VARS = grid_vars
        self.__UNKS = unks
        
    def __parse_from_directives(self, routine_file) -> dict:
        variables = {}
        json_string = ""
        for line in routine_file:
            if self.__DELIMITER in line:
                json_string += line[line.find(self.__DELIMITER)+len(self.__DELIMITER):].strip()
        json_string = "{" + json_string + "}"
        self._logger.log(self._TOOL_NAME, json_string, LOG_LEVEL_BASIC_DEBUG)
        variables = json.loads(json_string)
        return variables
    
    def __parse_from_code(self, routine_file) -> dict:
        # TODO: How to get all grid data units?
        # TODO: How to get all possible UNKS?
        read_write_mappings = {}

        current_subroutine = None
        full_line = ""
        for line in routine_file:

            if "subroutine" in line and not "end subroutine" in line:
                name_and_args = line[line.find("subroutine") + len("subroutine"):]
                name_and_args = name_and_args.split('(')
                assert len(name_and_args) == 2
                name = name_and_args[0]
                args = name_and_args[1].replace(')', '').split(',')
                current_subroutine = name.strip()
                read_write_mappings[current_subroutine] = {}
                for variable in self.__GRID_VARS:
                    read_write_mappings[current_subroutine].update( **{variable: {"R": set(), "W": set(), "RW": set()}} ) 
                continue

            full_line += ' '.join(line.split()) + '\n'
            if line.endswith("&\n"):
                full_line = full_line.replace("&\n", '')
                continue
            elif full_line.endswith("\n"):
                # print(full_line)
                # check line for variables
                # TODO: THis does not work if multiple equal statements on one line (Loops!)
                two_sides = full_line.split("=")
                if len(two_sides) == 2:
                    left = two_sides[0]
                    right = two_sides[1]
                    
                    for variable in self.__GRID_VARS:
                        write = set()
                        read = set()

                        # TODO: Potential security risk allowing raw variable into a regex pattern.
                        regex_string = self.__FIND_RW_EXPR_START + variable + self.__FIND_RW_EXPR_END
                        # match pattern in line
                        left_matches = re.findall(regex_string, left)
                        right_matches = re.findall(regex_string, right)

                        # check reads and writes
                        for unk in self.__UNKS:
                            for match in left_matches:
                                if unk in match:
                                    write.add(unk)
                            for match in right_matches:
                                if unk in match:
                                    read.add(unk)

                        read_write_mappings[current_subroutine][variable]["R"] = read_write_mappings[current_subroutine][variable]["R"].union(read)
                        read_write_mappings[current_subroutine][variable]["W"] = read_write_mappings[current_subroutine][variable]["W"].union(write)
                full_line = ""

        for routine in read_write_mappings.keys():
            for var in read_write_mappings[routine]:
                read = read_write_mappings[routine][var]["R"]
                write = read_write_mappings[routine][var]["W"]
                
                read_write_mappings[routine][var]["RW"] = read.intersection(write)
                read_write_mappings[routine][var]["R"] = read.difference(write)
                read_write_mappings[routine][var]["W"] = write.difference(read)

        # self._logger.log(self._TOOL_NAME, json.dumps(read_write_mappings, indent=4, default=serialize_sets), LOG_LEVEL_BASIC_DEBUG) 
        return read_write_mappings

    def parse_routine(self, routine_file) -> dict:
        return self.__parse_from_code(routine_file)
        # return self.__parse_from_directives(routine_file)
    
# SO solution
def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj
