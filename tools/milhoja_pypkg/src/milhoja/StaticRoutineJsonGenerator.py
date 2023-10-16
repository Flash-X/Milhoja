import json
import os

from pathlib import Path
from . import CURRENT_MILHOJA_JSON_VERSION
from . import BasicLogger

class StaticRoutineJsonGenerator():
    """
    This class cannot inherit from AbcCodeGenerator because
    it does not use a TaskFunction class.

    For now, this class will just read from directives and import them into 
    a json, but eventually will be able to parse files and get information into 
    a json without the json directives contained within the file.
    """

    def __init__(destination: str, files_to_parse: list, combine: bool, log_level, ):
        """
        Constructor. 

        :param str destination: The desination folder of the json outputs.
        :param list files_to_parse: The static fortran routine paths to parse.
        :param bool combine: Combines all json outputs into 1 json.
        """ 
        self.__TOOL_NAME = "StaticRoutineJsonGenerator"

        # convert strings to paths.
        self.__destination = os.path.expanduser(Path(destination).resolve())
        self.__files = [ os.path.expanduser(Path(file).resolve()) for file in files_to_parse ]
        self.__combine = combine 

        # set up logger
        

    def generate_files(self):
        """
        Generates jsons from the given file list in the constructor.
        """
        for file in self.__files:
            ...

    def combine(self):
        ...

    def parse_file(self, routine_file, ):
        ...


    
    
