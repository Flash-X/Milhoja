import src.main
import src.node

def initializeCodeGenerator():
    return src.main.initialize()

def finalizeCodeGenerator(basename=None):
    src.main.finalizeDeviceSetup()
    ttCode = src.main.generateThreadTeamCode()
    if basename is not None and ttCode is not None:
        with open(basename+'_threadTeam.ini', 'w') as f:
            f.write(ttCode)
    return src.main.finalize()

def Iterator(iterType : str):
    obj = src.node.IteratorNode(iterType)
    return src.main.addNodeAndLink(None, obj)

def ConcurrentDataBegin(unkIn=[], scratch=[]):
    obj = src.node.ConcurrentDataBeginNode(unkIn, scratch)
    def linkfn(sourceNode):  # depends on `obj`
        assert sourceNode is not None
        return src.main.addNodeAndLink(sourceNode, obj)
    return linkfn

def ConcurrentDataEnd(unkOut, **kwargs):
    obj = src.node.ConcurrentDataEndNode(unkOut)
    def linkfn(sourceNode):  # depends on `obj`
        assert sourceNode is not None
        return src.main.addNodeAndLink(sourceNode, obj)
    return linkfn

def Action(name : str, args=dict()):
    obj = src.node.ActionNode(name, args)
    def linkfn(sourceNode):  # depends on `obj`
        assert sourceNode is not None
        return src.main.addNodeAndLink(sourceNode, obj)
    return linkfn

def ConcurrentHardware(**kwargs):
    for device, info in kwargs.items():
        nodes = info['actions']
        src.main.setDevice(nodes, device)
        for key, value in info.items():
            if 'actions' == key:
                continue
            src.main.setAttribute(nodes, key, value)
    return
