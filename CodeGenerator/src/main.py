import src.graph
import src.runtime_specific
import networkx, copy, sys
import matplotlib.pyplot

###########
# Constants
###########

_PREFIX = '[CodeGenerator]'

_PLOTID_GRAPH            = 311
_PLOTID_H_GRAPH          = 312
_PLOTID_THREADTEAM_GRAPH = 313

##################
# Global Variables
##################

_graph = None
_h_graph = None
_threadteam_graph = None

###########
# Interface
###########

def initialize(codeAssembler):
    print(_PREFIX, 'Initialize')
    global _graph
    _graph = src.graph.TaskGraph(codeAssembler=codeAssembler, verbose=True)
    return _graph

def finalize():
    print(_PREFIX, 'Finalize')
    # plot
    print(_PREFIX, 'Control Flow Graph ({})'.format(_graph is not None),
          'Coarse Control Flow Graph ({})'.format(_h_graph is not None),
          'Thread Team Control Flow Graph ({})'.format(_threadteam_graph is not None))
    fig = matplotlib.pyplot.figure(figsize=(16,6))
    if _graph is not None:
        ax = matplotlib.pyplot.subplot(_PLOTID_GRAPH)
        ax.set_title('Control Flow Graph')
        _graph.plot(nodeLabels=True)
    if _h_graph is not None:
        ax = matplotlib.pyplot.subplot(_PLOTID_H_GRAPH)
        ax.set_title('Coarse Control Flow Graph')
        _h_graph.plot(nodeLabels=True)
    if _threadteam_graph is not None:
        ax = matplotlib.pyplot.subplot(_PLOTID_THREADTEAM_GRAPH)
        ax.set_title('Thread Team Control Flow Graph')
        pos_nodes  = src.graph.AbstractGraph.linear_layout(_threadteam_graph)
        networkx.draw_networkx_nodes(_threadteam_graph, pos_nodes, node_size=600)
        networkx.draw_networkx_edges(_threadteam_graph, pos_nodes,
                                     min_source_margin=15,
                                     min_target_margin=15)
        networkx.draw_networkx_labels(_threadteam_graph, pos_nodes, font_size=10)
    fig.set_tight_layout({'pad': 0.5})
    fig.savefig('_graphs.pdf')
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

def parseCode():
    _createThreadTeamGraph()
    assert _h_graph is not None
    # return if noting to do TODO deprecated
    if _threadteam_graph is None:
        return None
    # parse code
    assert 'CodeModelClass' in _h_graph.G.graph
    return _h_graph.parseCode()

####################
# Thread Team Graphs
####################

def _createThreadTeamGraph():
    assert _graph is not None
    global _h_graph
    global _threadteam_graph
    _h_graph          = _graph.toHierarchicalGraph()  # hierarchical task graph
    ttGraphList       = _createThreadTeamGraphList()   # thread team graphs
    _threadteam_graph = _searchThreadTeamGraph(ttGraphList, _h_graph.G)
    if _threadteam_graph is not None:
        print(_PREFIX, 'Matching Thread Team:', _threadteam_graph.graph['name'])
    else:
        print(_PREFIX, 'Matching Thread Team was not found.')
    return

def _createThreadTeamGraphList():
    ttGraphList = []
    for fn in src.runtime_specific.threadTeamGraphFunctions:
        ttGraphList.append(fn())
    return ttGraphList

def _searchThreadTeamGraph(ttGraphList, chkG):
    assert not ('CodeModelClass' in chkG.graph)
    for refG in ttGraphList:
        if _compareThreadTeamGraphs(refG, chkG):
            chkG.graph['CodeModelClass'] = refG.graph['CodeModelClass']
            return refG
    return None

def _compareThreadTeamGraphs(refG, chkG):
    def nm(refNodeAttr, chkNodeAttr):
        # exit if attributes do not exist
        if bool(refNodeAttr) and not bool(chkNodeAttr):  # if only chk is without attributes
            return False
        # compare attributes
        for key, val in refNodeAttr.items():
            if key.startswith('_'):  #TODO deprecated
                # copy hidden attributes from reference  #TODO deprecated
                #chkNodeAttr[key] = refNodeAttr[key]  # TODO deprecated
                continue  #TODO deprecated
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

##################
# Thread Team Code TODO DEPRECATED
##################

#def _generateThreadTeamCode_node(ttGraph):
#    code = ''
#    nodeids = set()
#    # add code from nodes
#    for u in ttGraph.nodes:
#        substituteDict = dict()
#        # extract nom-hidden attributes of this node
#        for key, attr in ttGraph.nodes[u].items():
#            if not key.startswith('_'):
#                substituteDict['__'+key+'__'] = attr
#        # extract arguments
#        argsDict = dict()
#        # create code of this node
#        for key, attr in ttGraph.nodes[u].items():
#            if key.startswith('_args'):
#                label = key.split(':')[1]
#                value = trim(str(attr))
#                argsDict[label] = value
#        for key, attr in ttGraph.nodes[u].items():
#            if key.startswith('_code'):
#                nodeids.add(u)
#                label = key.split(':')[1]
#                value = substitute(trim(attr), substituteDict)
#                code += '[{}]\n'.format(label)
#                if label in argsDict:
#                    code += 'args = ' + argsDict[label] + '\n'
#                code += 'definition =\n' + value + '\n\n'
#    code = '# id: ' + str(nodeids) + '\n\n' + code
#    return code
#
#def _generateThreadTeamCode_graph(ttGraph):
#    code = ''
#    # extract arguments
#    argsDict = dict()
#    for key, attr in ttGraph.graph.items():
#        if key.startswith('_args'):
#            label = key.split(':')[1]
#            value = trim(str(attr))
#            argsDict[label] = value
#    # add code from graph
#    for key, attr in ttGraph.graph.items():
#        if key.startswith('_code'):
#            label = key.split(':')[1]
#            value = trim(attr)
#            code += '[{}]\n'.format(label)
#            if label in argsDict:
#                code += 'args = ' + argsDict[label] + '\n'
#            code += 'definition =\n' + value + '\n\n'
#    return code
#
#def substitute(string : str, subDict : dict):
#    for old, new in subDict.items():
#        if old in string:
#            string = string.replace(old, str(new))
#    return string
#
#def trim(docstring : str):
#    '''Handle indentation.
#    Source: https://www.python.org/dev/peps/pep-0257/
#    '''
#    if not docstring:
#        return ''
#    # convert tabs to spaces (following the normal Python rules)
#    # and split into a list of lines:
#    lines = docstring.expandtabs().splitlines()
#    # determine minimum indentation (first line doesn't count):
#    indent = sys.maxsize
#    for line in lines[1:]:
#        stripped = line.lstrip()
#        if stripped:
#            indent = min(indent, len(line) - len(stripped))
#    # remove indentation (first line is special):
#    trimmed = [lines[0].strip()]
#    if indent < sys.maxsize:
#        for line in lines[1:]:
#            trimmed.append(line[indent:].rstrip())
#    # strip off trailing and leading blank lines:
#    while trimmed and not trimmed[-1]:
#        trimmed.pop()
#    while trimmed and not trimmed[0]:
#        trimmed.pop(0)
#    # return a single string:
#    return '\n'.join(trimmed)
