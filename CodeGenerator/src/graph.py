from src.node import *
import networkx, copy, sys

################
# Abstract Graph
################

class AbstractGraph():
    def __init__(self, verbose=False, verbose_prefix='[AbstractGraph]'):
        ''' Creates a new graph. '''
        # set options
        self.verbose        = verbose
        self.verbose_prefix = verbose_prefix
        # initialize node id's
        self.rootid = 0     # unique id of root node
        self.leafid = None  # unique id of leaf node
        self.nodeid = 0     # counter for node id's (begins counting at root id)

    def _copy(self):
        return AbstractGraph(verbose=self.verbose, verbose_prefix=self.verbose_prefix)

    def _setGraph(self, nxGraph : networkx.DiGraph, nodeid : int):
        assert not hasattr(self, 'G')
        assert isinstance(nxGraph, networkx.DiGraph), type(nxGraph)
        assert isinstance(nodeid, int), type(nodeid)
        self.G      = nxGraph
        self.nodeid = nodeid
        return self.nodeid

    def __str__(self):
        assert hasattr(self, 'G')
        return str(self.__class__) + ': Nodes = ' + str(self.G.nodes)

    @staticmethod
    def _searchBounds(nxGraph):
        lPath = networkx.dag_longest_path(nxGraph)
        return (lPath[0], lPath[-1])

    def getBounds(self):
        assert hasattr(self, 'G')
        if self.leafid is None:  # if need to search leaf id
            bounds = self._searchBounds(self.G)
            assert self.rootid == bounds[0], (self.rootid, bounds[0])
            self.leafid = bounds[1]
        return (self.rootid, self.leafid)

    @staticmethod
    def _isValid_nxGraph(nxGraph, rootidRef=None):
        # check if graph is a DAG
        if not networkx.is_directed_acyclic_graph(nxGraph):
            return False
        # check if graph starts at rootid
        bounds = AbstractGraph._searchBounds(nxGraph)
        if rootidRef is not None and not (rootidRef == bounds[0]):
            return False
        # check if all nodes are on a path between root and end nodes
        simplePaths = networkx.all_simple_paths(nxGraph, bounds[0], bounds[1])
        nodesPaths  = set(*simplePaths)
        nodesGraph  = set(nxGraph.nodes)
        return (nodesPaths == nodesGraph)

    def isValid(self):
        ''' Checks if graph is valid.  This implies that
            - the graph is a directed acyclic graph
            - the graph begins at one and only one node with id == rootid
            - the graph ends at one and only one node
            - all nodes of the graph are on a path between root and end nodes
        '''
        # check if graph exists
        if not hasattr(self, 'G'):
            return False
        # check if NetworkX graph is valid
        return self._isValid_nxGraph(self.G, self.rootid)

    def addNode(self, nodeObject):
        # init graph
        if not hasattr(self, 'G'):
            self.G = networkx.DiGraph()
        # update node id / counter
        i = self.nodeid
        self.nodeid += 1
        # deactivate leaf id
        self.leafid = None
        # add node
        if self.verbose:
            print(self.verbose_prefix, 'Add node: {},'.format(i), nodeObject)
        self.G.add_node(i, obj=nodeObject)
        return i

    def addNodeAttribute(self, node, attributeName : str, attributeValue):
        assert isinstance(attributeName, str), type(attributeName)
        try:
            for u in node:
                assert isinstance(u, int), type(u)
                self.G.nodes[u][attributeName] = attributeValue
        except TypeError:
            assert isinstance(node, int), type(node)
            self.G.nodes[node][attributeName] = attributeValue
        except:
            raise

    def addEdge(self, nodeSource : int, nodeTarget : int):
        assert isinstance(nodeSource, int) or nodeSource is None, type(nodeSource)
        assert isinstance(nodeTarget, int), type(nodeTarget)
        if nodeSource is None:
            nodeSource = self.rootid
        if self.verbose:
            print(self.verbose_prefix, 'Add edge:', nodeSource, '->', nodeTarget)
        self.G.add_edge(nodeSource, nodeTarget)
        return (nodeSource, nodeTarget)

    def nodeAttributeChange(self, nodeSource : int, nodeTarget : int, nodeAttributeName : str, defaultValue=None):
        assert isinstance(nodeSource, int), type(nodeSource)
        assert isinstance(nodeTarget, int), type(nodeTarget)
        assert isinstance(nodeAttributeName, str), type(nodeAttributeName)
        if nodeAttributeName in self.G.nodes[nodeSource]:
            valSrc = self.G.nodes[nodeSource][nodeAttributeName]
        else:
            valSrc = defaultValue
        if nodeAttributeName in self.G.nodes[nodeTarget]:
            valTrg = self.G.nodes[nodeTarget][nodeAttributeName]
        else:
            valTrg = defaultValue
        return (bool(valSrc is None) != bool(valTrg is None)) or (valSrc != valTrg)

    def markEdgesWithAttributeChange(self, nodeAttributeName : str, edgeAttributeName : str):
        assert isinstance(nodeAttributeName, str), type(nodeAttributeName)
        assert isinstance(edgeAttributeName, str), type(edgeAttributeName)
        for u, v in self.G.edges:
            self.G[u][v][edgeAttributeName] = self.nodeAttributeChange(u, v, nodeAttributeName)
        return edgeAttributeName

    def _createSubgraph_NetworkX(self, subBounds : tuple):
        assert isinstance(subNodeBounds, tuple), type(subNodeBounds)
        assert len(subNodeBounds) == 2, len(subNodeBounds)
        assert isinstance(subNodeBounds[0], int), type(subNodeBounds[0])
        assert isinstance(subNodeBounds[1], int), type(subNodeBounds[1])
        simplePaths = networkx.all_simple_paths(self.G, subBounds[0], subBounds[1])
        subNodes    = set(*simplePaths)
        return self.G.subgraph(list(subNodes))

    #-------------
    # Pruned Graph
    #-------------

    def _toPrunedGraph_NetworkX(self, edgeAttributeName : str):
        ''' see also: https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.subgraph.html '''
        assert isinstance(edgeAttributeName, str), type(edgeAttributeName)
        cpGraph = networkx.DiGraph()  # copy of nodes/edges from full graph
        rmNodes = set()               # storage for removed nodes
        cpGraph.add_node(self.rootid, **self.G.nodes[self.rootid])
        # separate graph in nodes/edges to keep and to remove
        for u, nbrsdict in self.G.adjacency():
            keep = False
            for v, eattr in nbrsdict.items():
                assert edgeAttributeName in eattr
                if eattr[edgeAttributeName]:
                    keep = True
                    break
            if keep:
                # copy target nodes and edges
                for v, eattr in nbrsdict.items():
                    cpGraph.add_node(v, **self.G.nodes[v])
                    if u in cpGraph:
                        cpGraph.add_edge(u, v, **eattr)
            else:
                # store source node of removed edge
                rmNodes.add(u)
        # connect "loose" nodes in pruned graph
        for u in rmNodes:
            if not u in cpGraph:
                continue
            targets = []
            candidates = list(self.G.successors(u))
            while candidates:  # while candidates remain
                v = candidates.pop()
                if v in cpGraph:
                    targets.append(v)
                else:
                    candidates.extend(list(self.G.successors(v)))
            for v in targets:
                cpGraph.add_edge(u, v)
        return cpGraph

    def toPrunedGraph(self, edgeAttributeName : str):
        if self.verbose:
            print(self.verbose_prefix, 'Create pruned graph based on edge attribute:', edgeAttributeName)
        graph = self._copy()
        graph._setGraph(self._toPrunedGraph_NetworkX(edgeAttributeName), self.nodeid)
        return graph

    #-------------------
    # Hierarchical Graph
    #-------------------

    def _toHierarchicalGraph_separateEdges(self, edgeAttributeName : str):
        assert isinstance(edgeAttributeName, str), type(edgeAttributeName)
        # separate edges based on the edge attribute
        coarseEdges = list()  # storage for edges of coarse graph
        subEdges    = list()  # storage for edges of (fine) subgraphs
        for u, adjdict in self.G.adjacency():
            isCoarseEdge = False
            adjitems = adjdict.items()
            for v, eattr in adjitems:
                if edgeAttributeName in eattr and eattr[edgeAttributeName]:
                    isCoarseEdge = True
                    break
            if isCoarseEdge:
                for v, eattr in adjitems:
                    coarseEdges.append((u, v))
            else:
                for v, eattr in adjitems:
                    subEdges.append((u, v))
        return coarseEdges, subEdges

    def _toHierarchicalGraph_createSubGraphs(self, subEdges : list):
        assert isinstance(subEdges, list), type(subEdges)
        # extract nodes of connected components
        edgeGraph = self.G.edge_subgraph(subEdges)
        connNodes = networkx.weakly_connected_components(edgeGraph)
        allNodes  = set(self.G.nodes)
        # create subgraphs
        subGraphs = list()
        for nodes in connNodes:  # loop over sets of nodes
            allNodes = allNodes.difference(nodes)
            subGraphs.append(self.G.subgraph(nodes).copy())
            assert self._isValid_nxGraph(subGraphs[-1])
        for node in allNodes:  # loop over individual nodes
            subGraphs.append(self.G.subgraph(node).copy())
        return subGraphs

    def _toHierarchicalGraph_createHierarchicalGraph(self, subGraphs : list, coarseEdges : list):
        assert isinstance(subGraphs, list), type(subGraphs)
        assert isinstance(coarseEdges, list), type(coarseEdges)
        # init hierarchical graph as a shallow copy of `self`
        hGraph = self._copy()
        # add subgraphs as nodes in hierarchical graph; get bounds of all subgraphs
        boundsSG = dict()
        for SG in subGraphs:
            node = hGraph.addNode(SG)
            boundsSG[node] = self._searchBounds(SG)
            for attributeName, attributeValue in self.G.nodes[boundsSG[node][0]].items():
                if attributeName in hGraph.G.nodes[node]:
                    continue
                hGraph.addNodeAttribute(node, attributeName, attributeValue)
        # translate edges for hierarchical graph
        hEdges = list()
        for edge in coarseEdges:  # loop over all coarse edges
            nodeSource = nodeTarget = hGraph.rootid - 1
            for i, b in boundsSG.items():  # loop over all subgraph bounds
                if nodeSource < hGraph.rootid and edge[0] == b[1]:
                    nodeSource = i
                if nodeTarget < hGraph.rootid and edge[1] == b[0]:
                    nodeTarget = i
            assert hGraph.rootid <= nodeSource and hGraph.rootid <= nodeTarget
            hEdges.append((nodeSource, nodeTarget))
        # link subgraphs / coarse nodes
        for j, edge in enumerate(hEdges):
            hGraph.addEdge(*edge)
            for attributeName, attributeValue in self.G.edges[coarseEdges[j]].items():
                hGraph.G.edges[edge][attributeName] = attributeValue
        return hGraph

    def toHierarchicalGraph(self, edgeAttributeName : str):
        if self.verbose:
            print(self.verbose_prefix, 'Create hierarchical graph based on edge attribute:', edgeAttributeName)
        coarseEdges, subEdges = self._toHierarchicalGraph_separateEdges(edgeAttributeName)
        subGraphs             = self._toHierarchicalGraph_createSubGraphs(subEdges)
        return                  self._toHierarchicalGraph_createHierarchicalGraph(subGraphs, coarseEdges)

    #---------
    # Plotting
    #---------

    def plot(self, nodeLabels=False, edgeLabels=False):
        pos_nodes  = networkx.circular_layout(self.G)
        pos_sublabels = copy.deepcopy(pos_nodes)
        magn = 0.0
        for pos in pos_sublabels.values():
            magn = max(magn, abs(pos[1]))
        for key in pos_sublabels.keys():
            pos_sublabels[key][1] -= 0.2*magn
#       plot_options = {
#           'with_labels': True,
#           'node_size': 600,
#           'font_size': 8,
#       #   'font_weight': 'bold',
#       #   'labels': networkx.get_node_attributes(self.G, 'device'),
#       }
#       networkx.draw(self.G, pos_nodes, **plot_options)
#      #networkx.draw_networkx_edge_labels(self.G, pos_nodes)
        networkx.draw_networkx_nodes(self.G, pos_nodes, node_size=600)
        networkx.draw_networkx_edges(self.G, pos_nodes,
                                     min_source_margin=15,
                                     min_target_margin=15)
        networkx.draw_networkx_labels(self.G, pos_nodes, font_size=10)
        labels_device = networkx.get_node_attributes(self.G, 'device')
        if nodeLabels:
            networkx.draw_networkx_labels(self.G, pos_sublabels, labels=labels_device, font_size=8)
        if edgeLabels:
            networkx.draw_networkx_edge_labels(self.G, pos_nodes, font_size=8)

############
# Task Graph
############

class RootNode():
    def __init__(self):
        self.type = 'Root'

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)

    def instantiateCodeAssembler(self, functionName=None, device=None):
        return None


class TaskGraph(AbstractGraph):
    def __init__(self, devices=['CPU', 'GPU'], verbose=False, initGraph=True):
        super().__init__(verbose=verbose, verbose_prefix='[TaskGraph]')
        if initGraph:
            self.addNode(RootNode())
        # set devices
        self.deviceDefault = 'Default'
        assert isinstance(devices, list), type(devices)
        if initGraph:
            assert not (self.deviceDefault in devices)
            self.deviceList = [self.deviceDefault, *devices]
        else:
            assert (self.deviceDefault in devices)
            self.deviceList = devices
        # set attribute names pertaining to devices
        self.deviceName       = 'device'
        self.deviceChangeName = self.deviceName + '_change'
        self.deviceSourceName = self.deviceName + '_source'
        self.deviceTargetName = self.deviceName + '_target'
        # set attribute names pertaining to memory
        self.memoryName       = 'memory'
        self.memoryCopy       = self.memoryName + '_copy'
        self.memoryScratch    = self.memoryName + '_scratch'

    def _copy(self):
        shallowCopy = TaskGraph(devices=self.deviceList, verbose=self.verbose, initGraph=False)
        copyAttr = [
            'deviceDefault',
            'deviceName', 'deviceChangeName', 'deviceSourceName', 'deviceTargetName',
            'memoryName', 'memoryCopy', 'memoryScratch']
        for name in copyAttr:
            setattr(shallowCopy, name, getattr(self, name))
        return shallowCopy

    def setNodeDevice(self, node, device : str):
        assert device in self.deviceList, device
        return self.addNodeAttribute(node, self.deviceName, device)

    def _setUpEdgesRecurively(self, node : int, memoryCopy : list, memoryScratch : list):
        '''
            node            Current node
            memoryCopy      Memory to copy for action corresponding to current and adjacent nodes
            memoryScratch   Scratch memory required for current and adjacent nodes
        '''
        assert isinstance(node, int), type(node)
        assert isinstance(memoryCopy, list), type(memoryCopy)
        assert isinstance(memoryScratch, list), type(memoryScratch)
        # process (current) node
        if self.deviceName not in self.G.nodes[node]:  # if device is not set
            self.G.nodes[node][self.deviceName] = self.deviceDefault
        assert 'obj' in self.G.nodes[node]
        if isinstance(self.G.nodes[node]['obj'], ConcurrentDataBeginNode):  # if has memory information
            memoryCopy.extend(self.G.nodes[node]['obj'].Uin)
            memoryScratch.extend(self.G.nodes[node]['obj'].scratch)
        # check the node's neighbors
        for nbr in list(self.G.successors(node)):  # loop over all neighbors
            # process neighboring node
            if self.deviceName not in self.G.nodes[nbr]:  # if device is not set
                self.G.nodes[nbr][self.deviceName] = self.deviceDefault
            isChange = self.G[node][nbr][self.deviceChangeName] = \
                self.nodeAttributeChange(node, nbr, self.deviceName)
            # set edge attributes pertaining to device change
            if isChange:
                self.G[node][nbr][self.deviceSourceName] = self.G.nodes[node][self.deviceName]
                self.G[node][nbr][self.deviceTargetName] = self.G.nodes[nbr][self.deviceName]
                self.G[node][nbr][self.memoryCopy]       = memoryCopy
                self.G[node][nbr][self.memoryScratch]    = memoryScratch
            # recurse into neighboring node
            self._setUpEdgesRecurively(nbr, memoryCopy, memoryScratch)
        return

    def setUp(self):
        self._setUpEdgesRecurively(node=self.rootid, memoryCopy=list(), memoryScratch=list())
        return

    def toPrunedGraph(self):
        return super().toPrunedGraph(self.deviceChangeName)

    def toHierarchicalGraph(self):
            return super().toHierarchicalGraph(self.deviceChangeName)

    def parseCode(self):
        return self.parseCode_nxGraph(self.G, self.rootid)

    @staticmethod
    def parseCode_nxGraph(nxGraph, rootid=0):
        subroutine = dict()
        codeNodes  = set()  #TODO unused
        code = TaskGraph._parseCodeRecurively_nxGraph(nxGraph, rootid, subroutine, codeNodes)
        return code, subroutine

    @staticmethod
    def _parseCodeRecurively_nxGraph(nxGraph, node : int, subroutine : dict, codeNodes : set):
        assert isinstance(node, int), type(node)
        assert isinstance(subroutine, dict), type(subroutine)
        assert isinstance(codeNodes, set), type(codeNodes)
        # extract (non-hidden) attributes of current node
        substituteDict = {'__ID__': node}  # TODO make params dict with `_param:` entries
        for key, attr in nxGraph.nodes[node].items():
            if not key.startswith('_'):
                substituteDict['__'+key+'__'] = attr
                #TODO update param variables
        # extract arguments
        argsDict = dict()
        for key, attr in nxGraph.nodes[node].items():
            if key.startswith('_args'):
                label = TaskGraph._substitute(key.split(':')[1], substituteDict)
                value = TaskGraph._trim(str(attr))
                argsDict[label] = value
        # process subgraph of (current) node
        assert 'obj' in nxGraph.nodes[node]
        subRoutineName = 'subroutine_id{}'.format(node)
        subGraph       = nxGraph.nodes[node]['obj']
        subCode        = TaskGraph._parseCode_subgraph(subGraph, subRoutineName)
        if subCode:
            substituteDict['__subroutine_name__'] = subRoutineName
            subroutine[subRoutineName] = subCode
        # process (current) node
        code = ''
        for key, attr in nxGraph.nodes[node].items():
            if key.startswith('_code'):
                label = TaskGraph._substitute(key.split(':')[1], substituteDict)
                value = TaskGraph._substitute(TaskGraph._trim(attr), substituteDict)
                code += '[{}]\n'.format(label)
                if label in argsDict:
                    code += 'args = ' + argsDict[label] + '\n'
                code += 'definition =\n' + value + '\n\n'
                codeNodes.add(node)
        # go to the node's neighbors
        for nbr in list(nxGraph.successors(node)):  # loop over all neighbors
            code += TaskGraph._parseCodeRecurively_nxGraph(nxGraph, nbr, subroutine, codeNodes)
        return code

    @staticmethod
    def _parseCode_subgraph(subGraph, subRoutineName):
        assert isinstance(subGraph, networkx.DiGraph), type(subGraph)
        assert isinstance(subRoutineName, str), type(subRoutineName)
        rootid = AbstractGraph._searchBounds(subGraph)[0]
        assert 'obj' in subGraph.nodes[rootid]
        assert 'device' in subGraph.nodes[rootid]
        device        = subGraph.nodes[rootid]['device']
        codeAssembler = subGraph.nodes[rootid]['obj'].instantiateCodeAssembler(functionName=subRoutineName, device=device)
        #TODO why is code assembler coming from a specific node?
        if codeAssembler:
            TaskGraph._parseCodeRecursively_subgraph(subGraph, rootid, codeAssembler)
            return codeAssembler.parse()
        else:
            return None

    @staticmethod
    def _parseCodeRecursively_subgraph(subGraph, node, codeAssembler):
        assert 'obj' in subGraph.nodes[node]
        assert 'device' in subGraph.nodes[node]
        # process (current) node
        device = subGraph.nodes[node]['device']
        subGraph.nodes[node]['obj'].assembleCode(codeAssembler, device=device)
        # go to the node's neighbors
        for nbr in list(subGraph.successors(node)):  # loop over all neighbors
            TaskGraph._parseCodeRecursively_subgraph(subGraph, nbr, codeAssembler)

    @staticmethod
    def _substitute(string : str, subDict : dict):
        for old, new in subDict.items():
            if old in string:
                string = string.replace(old, str(new))
        return string

    @staticmethod
    def _trim(docstring : str):
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
