#!/usr/bin/env python

from milhoja.FortranStaticRoutineParser import FortranStaticRoutineParser
from milhoja.FortranStaticRoutineJsonGenerator import FortranStaticRoutineJsonGenerator
from milhoja import LOG_LEVEL_BASIC_DEBUG

if __name__ == "__main__":
    files = [
            "/home/wkwiecinski/Flash-X/source/Simulation/SimulationMain/Sedov/Milhoja/Hydro_computeFluxes_X_block_cpu.F90",
    ]
    destination = "~/OrchestrationRuntime/tools"
    combine = False

    generator = FortranStaticRoutineParser(destination, files, LOG_LEVEL_BASIC_DEBUG, "$flashx")
    generator.generate_files()
