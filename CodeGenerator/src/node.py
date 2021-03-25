import copy, json, pathlib, re

################
# Code Assembler
################

class CodeAssembler():
    def __init__(self, codePath=None, indentSpace=' '*4, verbose=True, debug=False):
        self.indentSpace = indentSpace
        self.verbose     = verbose
        self.debug       = debug
        if codePath is not None:
            self.codePath = codePath
        else:
            self.codePath = pathlib.Path('.')

    @staticmethod
    def load(path, isTopLayer=False):
        with open(path, 'r') as f:
            tree = json.load(f)
            if isTopLayer:
                # remove connectors from top layer
                items     = tree.pop('_connector:setup', None)
                moreItems = tree.pop('_connector:execute', None)
                if items is not None and moreItems is not None:
                    for key, value in moreItems.items():
                        if '_code' == key:
                            try:
                                items[key].extend(value)
                            except KeyError:
                                items[key] = value
                            except:
                                raise
                        else:
                            items[key] = value
                elif moreItems is not None:
                    items = moreItems
                tree.update(items)
            return tree
        return None

    @staticmethod
    def dump(tree, path):
        with open(path, 'w') as f:
            json.dump(tree, f, indent=4)

    def instantiateCodeAssembler(self, functionName=None, device=None):
        assert isinstance(functionName, str), type(functionName)
        assert isinstance(device, str), type(device)
####DEV
        path_default = self.codePath / 'ex_sedov_subroutine_Default_main.json'
        path_device  = self.codePath / ('ex_sedov_subroutine_' + device + '_main.json')
####DEV
        self.tree = CodeAssembler.load(path_default, isTopLayer=True)
        loc = self.link(CodeAssembler.load(path_device))
        self.linkLocation = loc[-1]
        self.tree['_param:functionName'] = functionName
        self.device = device
        return self

    def link(self, treeLink, linkLocation=tuple()):
        assert isinstance(linkLocation, tuple), type(linkLocation)
        c = self.tree
        for l in linkLocation:
            c = c[l]
        locations =  CodeAssembler.link_trees(c, treeLink)
        return locations

    @staticmethod
    def link_trees(tree, treeLink):
        assert isinstance(tree, dict), type(tree)
        assert isinstance(treeLink, dict), type(treeLink)
        topLevelParams = dict()
        CodeAssembler._gather_parameters(treeLink, topLevelParams)
        locations = list()
        for label in ['setup', 'execute']:
            connLabel = '_connector:' + label
            linkLabel = '_link:' + label
            if connLabel in treeLink:
                c = copy.deepcopy(treeLink[connLabel])
                for key, value in topLevelParams.items():
                    if not key in c:
                        c[key] = value
                if '_code' in tree:
                    for i in range(len(tree['_code'])):  # loop over all code entries
                        if isinstance(tree['_code'][i], dict) and (linkLabel in tree['_code'][i]):
                            tree['_code'][i][linkLabel].append(c)
                            j = len(tree['_code'][i][linkLabel]) - 1
                            locations.append( ('_code', i, linkLabel, j) )
        return locations

    def parse(self):
        # write debug file
        if self.debug:
            cl = self.__class__.__name__
            fn = '{}_{}'.format(self.tree['_param:functionName'], self.device)
            CodeAssembler.dump(self.tree, '_debug_{}_{}.json'.format(cl, fn))
        # parse tree
        lines  = list()
        params = dict()
        CodeAssembler.parse_tree(self.tree, lines, params, self.indentSpace, 0, self.verbose)
        # parse lines
        return '\n'.join(lines)

    @staticmethod
    def parse_tree(tree, lines, params, indentSpace, indentCount, verbose):
        assert isinstance(tree, dict), type(tree)
        assert isinstance(lines, list), type(lines)
        assert isinstance(params, dict), type(params)
        # gather parameters
        indentCount += CodeAssembler._gather_parameters(tree, params)
        # process links
        for linkLabel in ['_link:setup', '_link:execute']:
            if linkLabel in tree:
                assert not ('_code' in tree)
                for ld in tree[linkLabel]:
                    CodeAssembler.parse_tree(ld, lines, copy.deepcopy(params), indentSpace, indentCount, verbose)
        # set indentation
        indent = indentSpace * indentCount
        # process code
        if '_code' in tree:
            assert not ('_link:setup' in tree or '_link:execute' in tree)
            if verbose and '_param:__file__' in tree:
                lines.append(indent + r'/* <' + tree['_param:__file__'] + '> */')
            for c in tree['_code']:
                if isinstance(c, str):
                    lines.append(indent + CodeAssembler._substitute_parameters(c, params))
                elif isinstance(c, dict):
                    CodeAssembler.parse_tree(c, lines, copy.deepcopy(params), indentSpace, indentCount, verbose)
                else:
                    raise TypeError('Expected str or dict, but got {}'.format(type(c)))
            if verbose and '_param:__file__' in tree:
                lines.append(indent + r'/* </' + tree['_param:__file__'] + '> */')

    @staticmethod
    def _gather_parameters(tree, params):
        indentCount = 0
        for key, value in tree.items():
            if key.startswith('_param:'):
                if '_param:indent' == key:
                    indentCount += value
                else:
                    params[key] = value
        return indentCount

    @staticmethod
    def _substitute_parameters(line, params):
        assert isinstance(line, str), type(line)
        assert isinstance(params, dict), type(params)
        if not ('_param:' in line):
            return line
        for key, value in params.items():
            line = re.sub(r'\b'+str(key)+r'\b', str(value), line)
        return line

#####################
# Nodes of Task Graph
#####################

class AbstractNode():
    def __init__(self, nodeType='Abstract'):
        self.type = nodeType

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class IteratorNode(AbstractNode):
    def __init__(self, iterType : str):
        super().__init__(nodeType='Iterator')
        assert isinstance(iterType, str), type(iterType)
        self.iterType = iterType

    def instantiateCodeAssembler(self, functionName=None, device=None):
        return None


class ConcurrentDataBeginNode(AbstractNode):
    def __init__(self, Uin : list, scratch : list):
        super().__init__(nodeType='ConcurrentDataBegin')
        assert isinstance(Uin, list), type(Uin)
        assert isinstance(scratch, list), type(scratch)
        self.Uin     = Uin
        self.scratch = scratch

    def instantiateCodeAssembler(self, functionName=None, device=None):
        return None


class ConcurrentDataEndNode(AbstractNode):
    def __init__(self, Uout : list):
        super().__init__(nodeType='ConcurrentDataEnd')
        assert isinstance(Uout, list), type(Uout)
        self.Uout = Uout

    def instantiateCodeAssembler(self, functionName=None, device=None):
        return None


class ActionNode(AbstractNode):
####DEV
    _codePath = pathlib.Path('../example')
    _codeData = CodeAssembler.load(_codePath / 'ex_sedov_Default_data.json')
####DEV

    def __init__(self, name : str, args : list):
        super().__init__(nodeType='Action')
        assert isinstance(name, str), type(name)
        assert isinstance(args, list), type(args)
        self.name = name
        self.args = args
        assert all([('_param:{}'.format(a) in self._codeData) for a in self.args])

    def instantiateCodeAssembler(self, functionName=None, device=None):
        codeAssembler = CodeAssembler(codePath=self._codePath)
        codeAssembler.instantiateCodeAssembler(functionName, device)
        return codeAssembler

    def assembleCode(self, codeAssembler, device=None):
        # create function code
        fnArgs = ['_param:{}'.format(a) for a in self.args]
        if device is not None:
            name = '{}_{}'.format(self.name, device)
        else:
            name = '{}'.format(self.name)
        fnCode = [name + '(' + ', '.join(fnArgs) + ');']
        fnConnector = { '_connector:execute': {'_code': fnCode} }
        # embed function code into a kernel
####DEV
        assert isinstance(device, str), type(device)
        assert codeAssembler.device == device
        path_kernel = self._codePath / ('ex_sedov_subroutine_' + device + '_action_kernel.json')
####DEV
        tree_kernel = codeAssembler.load(path_kernel)
        codeAssembler.link_trees(tree_kernel['_connector:execute'], fnConnector)
        # setup arguments
        argsCode = list()
        for a in self.args:
            s = '_param:setup_{}'.format(a)
            if s in tree_kernel['_connector:execute']:
                c = tree_kernel['_connector:execute'][s]
                if isinstance(c, str):
                    argsCode.append(c)
                elif isinstance(c, list):
                    argsCode.extend(c)
                else:
                    raise TypeError('Expected str or list, but got {}'.format(type(c)))
        if argsCode:
            argsConnector = { '_connector:setup': {'_code': argsCode}}
            codeAssembler.link_trees(tree_kernel['_connector:execute'], argsConnector)
        # embed kernel into a loop
        if 'CPU' == device:
            tree = tree_kernel
        if 'GPU' == device:
####DEV
            path_loop = self._codePath / ('ex_sedov_subroutine_' + device + '_action_loop.json')
####DEV
            tree_loop = codeAssembler.load(path_loop)
            codeAssembler.link_trees(tree_loop['_connector:execute'], tree_kernel)
            tree = tree_loop
        # link function code into code tree
        locations = codeAssembler.link(tree, codeAssembler.linkLocation)
        if not locations:
            cl = self.__class__.__name__
            fn = 'assembleCode'
            codeAssembler.dump(self.tree, '_debug_{}_{}.json'.format(cl, fn))
        assert locations, 'Linking failed, using link location: ' + str(codeAssembler.linkLocation)
