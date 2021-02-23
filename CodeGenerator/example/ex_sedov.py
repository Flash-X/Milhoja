############
# Initialize
############

import sys

sys.path.append('..')
from CodeGenerator import *

BASENAME: str = 'ex_sedov'

initializeCodeGenerator()

########
# Recipe
########

it   = Iterator(iterType='leaves')
dIn  = ConcurrentDataBegin(unkIn=['DENS', 'VELX', 'VELY', 'VELZ', 'PRES', 'ENER'],
                           scratch=['FLX', 'FLY', 'FLZ', 'AUXC'])(it)
hyFl = Action(name='hy_computeFluxesHll')(dIn)
hyUp = Action(name='hy_updateSolutionHll')(hyFl)
ioIq = Action(name='io_computeIntegralQuantitiesByBlock')(hyUp)
dOut = ConcurrentDataEnd(unkOut=['DENS', 'VELX', 'VELY', 'VELZ', 'PRES', 'ENER'])(ioIq)

ConcurrentHardware(CPU={'nInitialThreads':  4, 'nTilesPerPacket':  0, 'actions': [ioIq]},
                   GPU={'nInitialThreads': 16, 'nTilesPerPacket': 64, 'actions': [hyFl, hyUp]})

# Note: dict key 'actions' is not a good name since the corresponding list
#       contains "action handles"
# TODO: allow to pass tuples (actionBegin, actionEnd) that define a subgraph to
#       ConcurrentHardware()

##########
# Finalize
##########

finalizeCodeGenerator(BASENAME)
