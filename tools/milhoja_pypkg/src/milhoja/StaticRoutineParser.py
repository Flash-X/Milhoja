import json
import os

from pathlib import Path
from . import CURRENT_MILHOJA_JSON_VERSION
from . import MILHOJA_JSON_FORMAT
from . import LOG_LEVEL_BASIC_DEBUG
from . import BasicLogger
from . import AbcCodeGenerator
from . import TaskFunction

class StaticRoutineParser():
    """
    This class should be able to be extended or inherited to include other languages

    For now, this class will just read from directives and import them into 
    a json, but eventually will be able to parse files and get information into 
    a json without the json directives contained within the file.
    """

    def __init__(self, destination: str, files_to_parse: list, log_level):
        """
        Constructor. 

        :param str destination: The desination folder of the json outputs.
        :param list files_to_parse: The static fortran routine paths to parse.
        :param bool combine: Combines all json outputs into 1 json.

        """ 
        self._TOOL_NAME = self.__class__.__name__
        self._logger = BasicLogger(log_level)
        self._logger.log(self._TOOL_NAME, "Setting up StaticRoutineParser...", LOG_LEVEL_BASIC_DEBUG)

        # convert strings to paths.
        self.__destination = Path(os.path.expanduser(destination)).resolve()
        self.__files = [
            Path(os.path.expanduser(file)).resolve()
            for file in files_to_parse
        ]

    def parse_routines(self) -> dict:
        """
        Generates jsons from the given file list in the constructor.

        :return: A dict containing all information needed from every routine.
        """
        routines = {}
        for file in self.__files:
            basename = os.path.basename(file)
            self._logger.log(self._TOOL_NAME, f"Parsing {basename}.", LOG_LEVEL_BASIC_DEBUG)

            if not file.is_file():
                self._logger.error(self._TOOL_NAME, f"{str(file)} is not a file!")
                exit(-1)

            with open(file) as routine_file:
                routines.update(self.parse_routine(routine_file))

        return routines

    def parse_routine(self, routine_file) -> dict:
        """
        Generates a json based on a given routine file.
        Base method returns an empty json with a given format. 
        Overridden methods should be parsing the routine file in some way.
        """
        raise NotImplementedError(f"parse_routine is not implemented for {self.__class__.__name__}.")

    


    
    
