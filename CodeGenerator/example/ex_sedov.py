############
# Initialize
############

import pathlib, sys

sys.path.append('..')
from codeflow.CodeGenerator import *
from codeflow.CodeAssembler import *

BASENAME: str = 'ex_sedov'

codeAssembler = CodeAssembler(codePath=pathlib.Path('.'), debug=True)
initializeCodeGenerator(codeAssembler)

########
# Recipe
########

it   = Iterator(iterType='leaves')
dIn  = ConcurrentDataBegin(Uin=['DENS', 'VELX', 'VELY', 'VELZ', 'PRES', 'ENER'],
                           scratch=['flX', 'flY', 'flZ', 'auxC'])(it)
hySp = Action(name='hy_computeSoundSpeedHll', args=['lo', 'hi', 'U', 'auxC'])(dIn)
#ConcurrentActionsBegin()  #TODO
hyFl = Action(name='hy_computeFluxesHll', args=['dt', 'lo', 'hi', 'deltas', 'U', 'flX', 'flY', 'flZ', 'auxC'])(hySp)
hyUp = Action(name='hy_updateSolutionHll', args=['lo', 'hi', 'U', 'flX', 'flY', 'flZ'])(hyFl)
#ConcurrentActionsEnd()  #TODO
#Fuse((hyFl, hyUp))  #TODO
#TODO rename Action arg. `name` to `routine`
eosG = Action(name='eos_idealGammaDensIe', args=['lo', 'hi', 'U'])(hyUp)
ioIq = Action(name='io_computeIntegralQuantitiesByBlock', args=['tId', 'lo', 'hi', 'volumes', 'U'])(eosG)
dOut = ConcurrentDataEnd(Uout=['DENS', 'VELX', 'VELY', 'VELZ', 'PRES', 'ENER'])(ioIq)

ConcurrentHardware(CPU={'nInitialThreads':  4, 'nTilesPerPacket':  0, 'actions': [ioIq]},
                   GPU={'nInitialThreads': 16, 'nTilesPerPacket': 64, 'actions': [hySp, hyFl, hyUp, eosG]})
#split action between CPU & GPU TODO

# Note: dict key 'actions' is not a good name since the corresponding list
#       contains "action handles"
# TODO: allow to pass tuples (actionBegin, actionEnd) that define a subgraph to
#       ConcurrentHardware()
# Q: do we need ConcurrentDataEnd?

##########
# Finalize
##########

finalizeCodeGenerator(BASENAME)
