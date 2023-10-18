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
        unks = {
            "VELX_VAR", "HY_XMOM_FLUX", "HY_YMOM_FLUX", "HY_ZMOM_FLUX", "DENS_VAR", 
            "HY_ENER_FLUX", "ENER_VAR", "VELY_VAR", "VELZ_VAR"
        }

        full_line = ""
        for line in routine_file:
            if full_line == "" or line.endswith("&\n"):
                full_line += ' '.join( line.replace("&\n", '').split() )
                continue
            else:
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
                        right_matches = re.findall(regex_string, right)
                        
                        # check reads and writes
                        for unk in unks:
                            # print("UNK: ", unk)
                            for match in left_matches:
                                # print("MATCH: ", match)
                                if unk in match:
                                    # print(f"Added {unk} to writes")
                                    write.add(unk)
                            for match in right_matches:
                                # print("MATCH: ", match)
                                if unk in match:
                                    # print(f"Added {unk} to reads")
                                    read.add(unk)

                        read_writes = read.intersection(write)
                        # read = read.difference(read_writes)
                        # write = write.difference(read_writes)

                        if variable not in read_write_mappings:
                            read_write_mappings[variable] = {"R": set(), "W": set(), "RW": set()}
                        # print(read, write, read_writes)
                        read_write_mappings[variable]["R"] = read_write_mappings[variable]["R"].union(read)
                        read_write_mappings[variable]["W"] = read_write_mappings[variable]["W"].union(write)
                        read_write_mappings[variable]["RW"] = read_write_mappings[variable]["RW"].union(read_writes)
                full_line = ""
        self._logger.log(self._TOOL_NAME, json.dumps(read_write_mappings, indent=4, default=serialize_sets), LOG_LEVEL_BASIC_DEBUG) 

    def parse_routine(self, routine_file) -> dict:
        self.__parse_from_code(routine_file)
        return self.__parse_from_directives(routine_file)
    
# SO solution
def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj
