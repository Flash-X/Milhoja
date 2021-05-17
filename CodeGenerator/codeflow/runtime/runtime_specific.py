#from src.graph import *
import networkx

#############
# Code Models
#############

class CodeModel():
    def __init__(self, codeAssembler):
        self.codeAssembler = codeAssembler

    def copyCodeAssembler(self):
        return self.codeAssembler.copy()

    def assembleCode_setup(self, uniqueId, name, **kwargs):
        pass

    def assembleCode_execute(self, uniqueId, name, **kwargs):
        pass

    def assembleCode(self, uniqueId, name, **kwargs):
        self.assembleCode_setup(uniqueId, name, **kwargs)
        self.assembleCode_execute(uniqueId, name, **kwargs)


class Driver_ExtendedGpuTasks(CodeModel):
    def __init__(self, codeAssembler):
        super().__init__(codeAssembler)

    def assembleCode_setup(self, uniqueId, name, **kwargs):
        assert 'nodeAttributes' in kwargs
        assert 'subRoutineName' in kwargs
        nodeAttributes = kwargs['nodeAttributes']
        subRoutineName = kwargs['subRoutineName']
        assert 'device' in nodeAttributes
        assert 'nInitialThreads' in nodeAttributes
        assert 'nTilesPerPacket' in nodeAttributes
        if 'GPU' == nodeAttributes['device']:
            teamType = 'ThreadTeamDataType::SET_OF_BLOCKS'
        else:
            teamType = 'ThreadTeamDataType::BLOCK'
        actionName = 'action_{}'.format(uniqueId)
        code = [
            'RuntimeAction  {};'.format(actionName),
            actionName + '.name            = "{}";'.format(uniqueId),
            actionName + '.nInitialThreads = {};'.format(nodeAttributes['nInitialThreads']),
            actionName + '.teamType        = {};'.format(teamType),
            actionName + '.nTilesPerPacket = {};'.format(nodeAttributes['nTilesPerPacket']),
            actionName + '.routine         = {};'.format(subRoutineName)
        ]
        tree = { '_connector:setup': {'_code': code} }
        # link code into code tree
        locations = self.codeAssembler.link(tree, self.codeAssembler.linkLocation)
        if not locations:
            cl = self.__class__.__name__
            fn = 'assembleCode_setup'
            self.codeAssembler.dump(tree, '_debug_{}_{}.json'.format(cl, fn))
        assert locations, 'Linking failed, using link location: ' + str(self.codeAssembler.linkLocation)
        return actionName

    def assembleCode_execute(self, uniqueId, name, **kwargs):
        assert 'action' in kwargs
        action = kwargs['action']
        assert isinstance(action, list), type(action)
        name = 'Action Pipeline'  #TODO more descriptive name
        code = [ '_param:runtime.executeExtendedGpuTasks("{}", {});'.format(name, ', '.join(action)) ]
        tree = { '_connector:execute': {'_code': code} }
        # link code into code tree
        locations = self.codeAssembler.link(tree, self.codeAssembler.linkLocation)
        if not locations:
            cl = self.__class__.__name__
            fn = 'assembleCode_execute'
            self.codeAssembler.dump(tree, '_debug_{}_{}.json'.format(cl, fn))
        assert locations, 'Linking failed, using link location: ' + str(self.codeAssembler.linkLocation)

####################
# Thread Team Graphs
####################

def createThreadTeamGraph_ExtendedGpuTasks():
    G = networkx.DiGraph(name='ExtendedGpuTasks')
    # add nodes
    G.add_node(0, device='Default')
    G.add_node(1, device='GPU', nInitialThreads=None, nTilesPerPacket=None)
    G.add_node(2, device='CPU', nInitialThreads=None, nTilesPerPacket=0)
    G.add_node(3, device='Default')
    # add edges
    networkx.add_path(G, [0, 1, 2, 3])
    # add code model
    G.graph['CodeModelClass'] = Driver_ExtendedGpuTasks
    # return thread team graph
    return G

# list of all functions that create a thread team graph
threadTeamGraphFunctions = [
    createThreadTeamGraph_ExtendedGpuTasks
    #TODO more thread team create functions
]
