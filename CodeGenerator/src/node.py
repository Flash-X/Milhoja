#####################
# Nodes of Task Graph
#####################

class RootNode():
    def __init__(self):
        self.type = 'Root'

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class IteratorNode():
    def __init__(self, iterType : str):
        self.type = 'Iterator'
        self.iterType = iterType

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class ConcurrentDataBeginNode():
    def __init__(self, unkIn : list, scratch : list):
        assert isinstance(unkIn, list), type(unkIn)
        assert isinstance(scratch, list), type(scratch)
        self.type = 'ConcurrentDataBegin'
        self.unkIn   = unkIn
        self.scratch = scratch

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class ConcurrentDataEndNode():
    def __init__(self, unkOut : list):
        assert isinstance(unkOut, list), type(unkOut)
        self.type = 'ConcurrentDataEnd'
        self.unkOut = unkOut

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class ActionNode():
    def __init__(self, name : str, args : dict):
        self.type = 'Action'
        self.name = name
        self.args = args

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)
