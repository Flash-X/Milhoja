#!/usr/bin/env python

from milhoja.CppStaticRoutineParser import CppStaticRoutineParser
from milhoja import LOG_LEVEL_BASIC_DEBUG

if __name__ == "__main__":
    files = [
       "~/OrchestrationRuntime/test/Base/computeLaplacianDensity.cpp"
    ]
    destination = "~/OrchestrationRuntime/tools"
    combine = False

    generator = CppStaticRoutineParser(destination, files, LOG_LEVEL_BASIC_DEBUG)
    generator.generate_files()
