################
# Pre-Processing
################

import sys

sys.path.append('..')
from CodeGenerator import *

initializeCodeGenerator()

########
# Recipe
########

it   = Iterator(iterType='leaves')
dIn  = ConcurrentDataBegin(unkIn=['U', 'V'], scratch=[])(it)
aQ   = Action(name='Q')(dIn)
aX   = Action(name='X', args={'pi': 3.16})(aQ)
aY   = Action(name='Y')(aX)
aZ   = Action(name='Z')(aX)
dOut = ConcurrentDataEnd(unkOut=['H', 'G'])([aY, aZ])

ConcurrentHardware(CPU={'nThreads':  4, 'actions': [aZ]},
                   GPU={'nThreads': 16, 'actions': [aQ, aX, aY]})

# Note: dict key 'actions' is not a good name since the corresponding list
#       contains "action handles"

#################
# Post-processing
#################

finalizeCodeGenerator()
