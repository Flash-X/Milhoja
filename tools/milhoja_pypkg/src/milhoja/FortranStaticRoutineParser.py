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

    def __init__(self, destination: str, files_to_parse: list, log_level, delimiter: str):
        """
        Constructor. 

        :param str destination: The desination folder of the json outputs.
        :param list files_to_parse: The static fortran routine paths to parse.
        """ 
        super().__init__(destination, files_to_parse, log_level)
        self.__DELIMITER = delimiter
        
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
    
    def __parse_from_code(self, routine_file):
        # TODO: How to get all grid data units?
        # TODO: How to get all possible UNKS?
        read_write_mappings = {}
        grid_data = {"U", "flX"}

        for variable in grid_data:
            read_write_mappings[variable] = {"R": set(), "W": set(), "RW": set()}

        unks = {
            "VELX_VAR", "HY_XMOM_FLUX", "HY_YMOM_FLUX", "HY_ZMOM_FLUX", "DENS_VAR", 
            "HY_ENER_FLUX", "ENER_VAR", "VELY_VAR", "VELZ_VAR", "HY_DENS_FLUX", "PRES_VAR"
        }

        full_line = ""
        for line in routine_file:
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
                    
                    for variable in grid_data:
                        write = set()
                        read = set()

                        # TODO: Potential security risk allowing raw variable into a regex pattern.
                        regex_string = self.__FIND_RW_EXPR_START + variable + self.__FIND_RW_EXPR_END
                        # match pattern in line
                        left_matches = re.findall(regex_string, left)
                        # print("Left: ", left_matches)
                        right_matches = re.findall(regex_string, right)
                        # print("Right: ", right_matches)

                        # check reads and writes
                        for unk in unks:
                            for match in left_matches:
                                if unk in match:
                                    write.add(unk)
                            for match in right_matches:
                                if unk in match:
                                    read.add(unk)

                        read_write_mappings[variable]["R"] = read_write_mappings[variable]["R"].union(read)
                        read_write_mappings[variable]["W"] = read_write_mappings[variable]["W"].union(write)
                full_line = ""

        for var in read_write_mappings:
            read = read_write_mappings[var]["R"]
            write = read_write_mappings[var]["W"]
            
            read_write_mappings[var]["RW"] = read.intersection(write)
            read_write_mappings[var]["R"] = read.difference(write)
            read_write_mappings[var]["W"] = write.difference(read)

        self._logger.log(self._TOOL_NAME, json.dumps(read_write_mappings, indent=4, default=serialize_sets), LOG_LEVEL_BASIC_DEBUG) 

    def parse_routine(self, routine_file) -> dict:
        self.__parse_from_code(routine_file)
        return self.__parse_from_directives(routine_file)
    
# SO solution
def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj
