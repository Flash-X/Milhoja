#!/usr/bin/env python

from milhoja.FortranStaticRoutineParser import FortranStaticRoutineParser
from milhoja import LOG_LEVEL_BASIC_DEBUG

if __name__ == "__main__":
    files = {
        "/home/wkwiecinski/Flash-X/source/Simulation/SimulationMain/Sedov/Milhoja/Hydro_computeFluxes_X_block_cpu.F90": 
        {
            "grid_vars": {"U", "flX"},
            "unks": {
                "VELX_VAR", "HY_XMOM_FLUX", "HY_YMOM_FLUX", "HY_ZMOM_FLUX", "DENS_VAR", 
                "HY_ENER_FLUX", "ENER_VAR", "VELY_VAR", "VELZ_VAR", "HY_DENS_FLUX", "PRES_VAR"
            }
        }
    }
    destination = "~/OrchestrationRuntime/tools"
    combine = False

    for file in files:
        generator = FortranStaticRoutineParser(
            destination, list(files.keys()), LOG_LEVEL_BASIC_DEBUG, "$flashx",
            files[file]["grid_vars"], files[file]["unks"]
        )
        print(generator.parse_routines())
        
