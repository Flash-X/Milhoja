import pathlib, sys

sys.path.append('..')
from CodeAssembler import *

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


class ConcurrentDataBeginNode(AbstractNode):
    def __init__(self, Uin : list, scratch : list):
        super().__init__(nodeType='ConcurrentDataBegin')
        assert isinstance(Uin, list), type(Uin)
        assert isinstance(scratch, list), type(scratch)
        self.Uin     = Uin
        self.scratch = scratch


class ConcurrentDataEndNode(AbstractNode):
    def __init__(self, Uout : list):
        super().__init__(nodeType='ConcurrentDataEnd')
        assert isinstance(Uout, list), type(Uout)
        self.Uout = Uout


class ActionNode(AbstractNode):
####DEV
    _codePath = pathlib.Path('../example')
    _codeData = CodeAssembler.load(_codePath / 'ex_sedov_data_Default.json')
####DEV

    def __init__(self, name : str, args : list):
        super().__init__(nodeType='Action')
        assert isinstance(name, str), type(name)
        assert isinstance(args, list), type(args)
        self.name = name
        self.args = args
        assert all([('_param:{}'.format(a) in self._codeData) for a in self.args])

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
            argsConnector = { '_connector:setup': {'_code': argsCode} }
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
            codeAssembler.dump(tree, '_debug_{}_{}.json'.format(cl, fn))
        assert locations, 'Linking failed, using link location: ' + str(codeAssembler.linkLocation)
