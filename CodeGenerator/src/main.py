from src.graph import *
import networkx, copy, sys
import matplotlib.pyplot

###########
# Constants
###########

_PREFIX = '[CodeGenerator]'

_PLOTID_GRAPH            = 131
_PLOTID_H_GRAPH          = 132
_PLOTID_THREADTEAM_GRAPH = 133

###########
# Interface
###########

_graph = None
_h_graph = None
_threadteam_graph = None

def initialize():
    print(_PREFIX, 'Initialize')
    global _graph
    _graph = TaskGraph(verbose=True)
    return _graph

def finalize():
    print(_PREFIX, 'Finalize')
    # plot
    print(_PREFIX, 'Plot TaskGraph ({})'.format(_graph is not None),
          'Coarse TaskGraph ({})'.format(_h_graph is not None),
          'Thread Team TaskGraph ({})'.format(_threadteam_graph is not None))
    fig = matplotlib.pyplot.figure(figsize=(16,6))
    if _graph is not None:
        ax = matplotlib.pyplot.subplot(_PLOTID_GRAPH)
        ax.set_title('TaskGraph')
        _graph.plot(nodeLabels=True)
    if _h_graph is not None:
        ax = matplotlib.pyplot.subplot(_PLOTID_H_GRAPH)
        ax.set_title('Coarse TaskGraph')
        _h_graph.plot(nodeLabels=True)
    if _threadteam_graph is not None:
        ax = matplotlib.pyplot.subplot(_PLOTID_THREADTEAM_GRAPH)
        ax.set_title('Thread Team TaskGraph')
        pos_nodes = networkx.circular_layout(_threadteam_graph)
        networkx.draw_networkx_nodes(_threadteam_graph, pos_nodes, node_size=600)
        networkx.draw_networkx_edges(_threadteam_graph, pos_nodes,
                                     min_source_margin=15,
                                     min_target_margin=15)
        networkx.draw_networkx_labels(_threadteam_graph, pos_nodes, font_size=10)
    fig.set_tight_layout({'pad': 0.5})
    matplotlib.pyplot.show()
    return

def addNodeAndLink(sourceNode, targetObject):
    assert _graph is not None
    assert targetObject is not None
    nodeid = _graph.addNode(targetObject)
    try:
        for src in sourceNode:
            _graph.addEdge(src, nodeid)
    except TypeError:
        assert isinstance(sourceNode, int) or sourceNode is None, type(sourceNode)
        _graph.addEdge(sourceNode, nodeid)
    return nodeid

def setAttribute(node, name : str, value):
    assert _graph is not None
    return _graph.addNodeAttribute(node, name, value)

def setDevice(node, device):
    assert _graph is not None
    return _graph.setNodeDevice(node, device)

def setUp():
    assert _graph is not None
    return _graph.setUp()

def _createThreadTeamGraph():
    assert _graph is not None
    global _h_graph
    global _threadteam_graph
    _h_graph          = _graph.toHierarchicalGraph()  # hierarchical task graph
    ttGraphList       = createThreadTeamGraphList()   # thread team graphs
    _threadteam_graph = matchThreadTeamGraph(ttGraphList, _h_graph.G)
    if _threadteam_graph is not None:
        print(_PREFIX, 'Matching Thread Team:', _threadteam_graph.graph['name'])
    else:
        print(_PREFIX, 'Matching Thread Team was not found.')
    return

def parseCode():
    _createThreadTeamGraph()
    assert _h_graph is not None
    # return if noting to do
    if _threadteam_graph is None:
        return None
    # parse code
    return _h_graph.parseCode()

##################
# Thread Team Code TODO DEPRECATED
##################

def _generateThreadTeamCode_node(ttGraph):
    code = ''
    nodeids = set()
    # add code from nodes
    for u in ttGraph.nodes:
        substituteDict = dict()
        # extract nom-hidden attributes of this node
        for key, attr in ttGraph.nodes[u].items():
            if not key.startswith('_'):
                substituteDict['__'+key+'__'] = attr
        # extract arguments
        argsDict = dict()
        # create code of this node
        for key, attr in ttGraph.nodes[u].items():
            if key.startswith('_args'):
                label = key.split(':')[1]
                value = trim(str(attr))
                argsDict[label] = value
        for key, attr in ttGraph.nodes[u].items():
            if key.startswith('_code'):
                nodeids.add(u)
                label = key.split(':')[1]
                value = substitute(trim(attr), substituteDict)
                code += '[{}]\n'.format(label)
                if label in argsDict:
                    code += 'args = ' + argsDict[label] + '\n'
                code += 'definition =\n' + value + '\n\n'
    code = '# id: ' + str(nodeids) + '\n\n' + code
    return code

def _generateThreadTeamCode_graph(ttGraph):
    code = ''
    # extract arguments
    argsDict = dict()
    for key, attr in ttGraph.graph.items():
        if key.startswith('_args'):
            label = key.split(':')[1]
            value = trim(str(attr))
            argsDict[label] = value
    # add code from graph
    for key, attr in ttGraph.graph.items():
        if key.startswith('_code'):
            label = key.split(':')[1]
            value = trim(attr)
            code += '[{}]\n'.format(label)
            if label in argsDict:
                code += 'args = ' + argsDict[label] + '\n'
            code += 'definition =\n' + value + '\n\n'
    return code

def substitute(string : str, subDict : dict):
    for old, new in subDict.items():
        if old in string:
            string = string.replace(old, str(new))
    return string

def trim(docstring : str):
    '''Handle indentation.
    Source: https://www.python.org/dev/peps/pep-0257/
    '''
    if not docstring:
        return ''
    # convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # return a single string:
    return '\n'.join(trimmed)

####################
# Thread Team Graphs
####################

def createThreadTeamGraphList():
    ttGraphList = []
    for fn in _threadTeamGraphFunctions:
        ttGraphList.append(fn())
    return ttGraphList

def matchThreadTeamGraph(ttGraphList, chkG):
    for refG in ttGraphList:
        if compareThreadTeamGraphs(refG, chkG):
            return refG
    return None

def compareThreadTeamGraphs(refG, chkG):
    def nm(refNodeAttr, chkNodeAttr):
        # exit if attributes do not exist
        if bool(refNodeAttr) and not bool(chkNodeAttr):  # if only chk is without attributes
            return False
        # compare attributes
        for key, val in refNodeAttr.items():
            if key.startswith('_'):
                # copy hidden attributes from reference
                chkNodeAttr[key] = refNodeAttr[key]
                continue
            if not (key in chkNodeAttr):
                return False
            if val is not None:
                if val != chkNodeAttr[key]:
                    return False
            else:
                # overwrite None value of reference
                refNodeAttr[key] = chkNodeAttr[key]
        return True
    return networkx.is_isomorphic(refG, chkG, node_match=nm, edge_match=None)
    #TODO create edge match function to copy over edge attributes

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
    # add code templates
    G.nodes[1]['_code:initActionBundle___ID__'] = \
        '''RuntimeAction actionBundle_GPU;'''
    G.nodes[1]['_code:setActionBundle___ID__'] = \
        '''actionBundle_GPU.name            = "ActionBundle __ID__";
           actionBundle_GPU.nInitialThreads = __nInitialThreads__;
           actionBundle_GPU.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
           actionBundle_GPU.nTilesPerPacket = __nTilesPerPacket__;
           actionBundle_GPU.routine         = __subroutine_name__;'''
    G.nodes[2]['_code:initActionBundle___ID__'] = \
        '''RuntimeAction actionBundle_CPU;'''
    G.nodes[2]['_code:setActionBundle___ID__'] = \
        '''actionBundle_CPU.name            = "ActionBundle __ID__";
           actionBundle_CPU.nInitialThreads = __nInitialThreads__;
           actionBundle_CPU.teamType        = ThreadTeamDataType::BLOCK;
           actionBundle_CPU.nTilesPerPacket = __nTilesPerPacket__;
           actionBundle_CPU.routine         = __subroutine_name__;'''
    G.nodes[3]['_args:executeActionBundles'] = \
        '''__runtime__'''
    G.nodes[3]['_code:executeActionBundles'] = \
        '''__runtime__.executeExtendedGpuTasks("ActionBundles GPU,CPU", actionBundle_GPU, actionBundle_CPU);'''
    # return thread team graph
    return G

# list of all functions that create a thread team graph
_threadTeamGraphFunctions = [
    createThreadTeamGraph_ExtendedGpuTasks
]
