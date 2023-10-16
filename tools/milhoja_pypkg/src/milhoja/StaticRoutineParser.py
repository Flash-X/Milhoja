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
    This class cannot inherit from AbcCodeGenerator because
    it does not use a TaskFunction class.

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

    def generate_files(self):
        """
        Generates jsons from the given file list in the constructor.
        """
        for file in self.__files:
            routine_json = None
            basename = os.path.basename(file)
            self._logger.log(self._TOOL_NAME, f"Parsing {basename}.", LOG_LEVEL_BASIC_DEBUG)

            if not file.is_file():
                self._logger.error(self._TOOL_NAME, f"{str(file)} is not a file!")
                exit(-1)

            with open(file) as routine_file:
                routine_json = self.parse_routine(routine_file)

            save_location = Path(self.__destination, basename.replace(file.suffix, '.json') ).resolve()
            self._logger.log(self._TOOL_NAME, f"Saving to {str(save_location)}", LOG_LEVEL_BASIC_DEBUG)
            with open(save_location, 'w') as fp:
                json.dump(routine_json, fp, indent=4)

    def parse_routine(self, routine_file) -> dict:
        """
        Generates a json based on a given routine file.
        Base method returns an empty json with a given format. 
        Overridden methods should be parsing the routine file in some way.
        """
        raise NotImplementedError(f"parse_routine is not implemented for {self.__class__.__name__}.")

    


    
    
