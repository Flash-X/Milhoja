NAME = 'Sedov via SourceTree'

import pathlib, sys
sys.path.append('../../')
from codeflow.assembler.sourcetree import SourceTree
from codeflow.assembler.sourcetree import load as sourcetree_load

def main():
    # initialize source tree with driver code
    assembler = SourceTree(codePath=pathlib.Path('.'), verbose=True, debug=False)
    assembler.initialize('template_driver.cpp')
    # add runtime actions
    params = {
        '_param:actionVariable':        'action_GPU',
        '_param:actionName':            'Sedov GPU action',
        '_param:nInitialThreads':       16,
        '_param:threadTeamDataType':    'ThreadTeamDataType::SET_OF_BLOCKS',
        '_param:nTilesPerPacket':       64,
        '_param:taskFunction':          'taskfunction_GPU'
    }
    assembler.link('template_runtimeAction.cpp', parameters=params)
    params = {
        '_param:actionVariable':        'action_CPU',
        '_param:actionName':            'Sedov CPU action',
        '_param:nInitialThreads':       4,
        '_param:threadTeamDataType':    'ThreadTeamDataType::BLOCK',
        '_param:nTilesPerPacket':       0,
        '_param:taskFunction':          'taskfunction_CPU'
    }
    assembler.link('template_runtimeAction.cpp', parameters=params)
    # add runtime execution
    treeLink = {
        '_connector:execute': {
            '_code': [
                '_param:runtime.executeExtendedGpuTasks("Sedov Pipeline", action_GPU, action_CPU);'
            ]
        }
    }
    assembler.link(treeLink)
    # parse driver code
    assembler.dump('_tree_example_sourcetree_driver.json')
    lines = assembler.parse()
    with open('_code_example_sourcetree_driver.cpp', 'w') as f:
        f.write(lines)

    # initialize source tree with a generic task function
    assembler_cpu = SourceTree(codePath=pathlib.Path('.'), verbose=True, debug=False)
    assembler_cpu.initialize('template_taskfunction_Default.json',
                             parameters={'_param:taskFunctionName': 'taskfunction_CPU'})
    # add CPU code
    linkLocation = assembler_cpu.link('template_taskfunction_CPU_main.json')
    # add CPU kernel
    tree_kernel = sourcetree_load('template_taskfunction_CPU_kernel.json')
    code_kernel = \
        tree_kernel['_param:setup_volumes'] + \
        ['io_computeIntegralQuantitiesByBlock_CPU(_param:dataItem, _param:lo, _param:hi, _param:volumes, _param:U);']
    loc = assembler_cpu.link(tree_kernel, linkLocation[-1])
    loc = assembler_cpu.link({ '_connector:execute': { '_code': code_kernel } }, linkLocation[-1])
    # parse CPU code
    assembler_cpu.dump('_tree_example_sourcetree_taskfunction_CPU.json')
    lines = assembler_cpu.parse()
    with open('_code_example_sourcetree_taskfunction_CPU.cpp', 'w') as f:
        f.write(lines)

    #TODO create GPU task function

if '__main__' == __name__:
    main()
