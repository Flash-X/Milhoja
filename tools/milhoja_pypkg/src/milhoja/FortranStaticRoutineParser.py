import json
import re

from .StaticRoutineParser import StaticRoutineParser
from . import LOG_LEVEL_BASIC_DEBUG
from . import MILHOJA_JSON_FORMAT
from . import CURRENT_MILHOJA_JSON_VERSION

# TODO: In order to get a list of unknown variables,
#       we need the list of all grid data structures being used,
#       and the list of all variable names. (VELX_VAR, VELY_VAR, etc.)

class FortranStaticRoutineParser(StaticRoutineParser):
    __FIND_RW_EXPR_START = r"(?<![^\s(_-])"
    __FIND_RW_EXPR_END = r"(?![^\s(_-])([^)]*)"
    __DEFAULT_DELIMITER = "$milhoja"

    __FORTRAN_TYPE_MAPPING = {
        "logical": "bool",
        "integer": "int",
        "real": "real"
    }

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

# TODO: Write a method for getting all read/write variables for each grid data structure.
    # def __check_read_write(self):
    #     full_line += ' '.join(line.split()) + '\n'
    #     if line.endswith("&\n"):
    #         full_line = full_line.replace("&\n", '')
    #         continue
    #     elif full_line.endswith("\n"):
    #         # print(full_line)
    #         # check line for variables
    #         # TODO: THis does not work if multiple equal statements on one line (Loops!)
    #         two_sides = full_line.split("=")
    #         if len(two_sides) == 2:
    #             left = two_sides[0]
    #             right = two_sides[1]
                
    #             for variable in self.__GRID_VARS:
    #                 write = set()
    #                 read = set()

    #                 # TODO: Potential security risk allowing raw variable into a regex pattern.
    #                 regex_string = self.__FIND_RW_EXPR_START + variable + self.__FIND_RW_EXPR_END
    #                 # match pattern in line
    #                 left_matches = re.findall(regex_string, left)
    #                 right_matches = re.findall(regex_string, right)

    #                 # check reads and writes
    #                 for unk in self.__UNKS:
    #                     for match in left_matches:
    #                         if unk in match:
    #                             write.add(unk)
    #                     for match in right_matches:
    #                         if unk in match:
    #                             read.add(unk)

    #                 read_write_mappings[current_subroutine][variable]["R"] = read_write_mappings[current_subroutine][variable]["R"].union(read)
    #                 read_write_mappings[current_subroutine][variable]["W"] = read_write_mappings[current_subroutine][variable]["W"].union(write)
    #         full_line = ""

    #     for routine in read_write_mappings.keys():
    #         for var in read_write_mappings[routine]:
    #             read = read_write_mappings[routine][var]["R"]
    #             write = read_write_mappings[routine][var]["W"]
                
    #             read_write_mappings[routine][var]["RW"] = read.intersection(write)
    #             read_write_mappings[routine][var]["R"] = read.difference(write)
    #             read_write_mappings[routine][var]["W"] = write.difference(read)
    #     return read_write_mappings

    # TODO: Allow directives to use multiple lines.
    def __parse_directive_statement(self, line):
        line = line[line.find(self.__DEFAULT_DELIMITER) + len(self.__DEFAULT_DELIMITER):].strip()
        key_and_arg = line.split("=")
        assert len(key_and_arg) == 2
        return { key_and_arg[0]: key_and_arg[1] }
    
    def __parse_json_information(self, routine_file) -> dict:
        # TODO: How to get all grid data units?
        # TODO: How to get all possible UNKS?
        pulling_subroutine_information = False
        routine_jsons = {}
        current_arg_spec = dict()
        used_delimiter = False

        current_subroutine = None
        full_line = ""
        for line in routine_file:

            # Collect subroutine information
            if "subroutine" in line and "end subroutine" not in line:
                name_and_args = line[line.find("subroutine") + len("subroutine"):]
                name_and_args = name_and_args.split('(')
                print(name_and_args)
                assert len(name_and_args) == 2
                name = name_and_args[0]
                args = [ arg.strip() for arg in name_and_args[1].replace(')', '').replace(' ', '').split(',') ]
                current_subroutine = name.strip()
                
                routine_jsons[current_subroutine] = {}
                routine_jsons[current_subroutine]["format"] = [MILHOJA_JSON_FORMAT, CURRENT_MILHOJA_JSON_VERSION]
                routine_jsons[current_subroutine]["interface_file"] = routine_file.name
                routine_jsons[current_subroutine]["argument_list"] = [ arg.replace(' ', '') for arg in args ]
                routine_jsons[current_subroutine]["argument_specifications"] = current_arg_spec

                if ")" not in line:
                    pulling_subroutine_information = True
                continue
            elif pulling_subroutine_information:
                if ")" in line:
                    pulling_subroutine_information = False
                    line = line.replace(')', '')
                args = line.strip().replace(' ', '').split(',')
                routine_jsons[current_subroutine]["argument_list"].append(args)
                continue
            elif self.__DEFAULT_DELIMITER in line:
                current_arg_spec.update(self.__parse_directive_statement(line))

                used_delimiter = True
            elif used_delimiter:
                # parse line containing variable declaration
                info_and_names = line.split("::")
                assert len(info_and_names) == 2

                info = info_and_names[0].strip().replace(' ', '')
                # fill current arg_spec with extra info

                # then get each name that uses that arg spec and insert into routine json.
                names = info_and_names[1].strip().replace(' ', '')
                
                used_delimiter = False
            elif "end subroutine" in line:
                current_subroutine = None
                continue
        return routine_jsons

    def parse_routine(self, routine_file) -> dict:
        return self.__parse_json_information(routine_file)
