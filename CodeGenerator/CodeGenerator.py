import src.main
import src.node

def initializeCodeGenerator():
    return src.main.initialize()

def finalizeCodeGenerator(basename=None):
    # set up and parse code
    src.main.setUp()
    code, subroutine = src.main.parseCode()
    # write main code
    if basename is not None and code is not None:
        with open('{}_main.ini'.format(basename), 'w') as f:
            f.write(code)
    # write subroutine code
    for nameSub, codeSub in subroutine.items():
        with open('{}_{}.cpp'.format(basename, nameSub), 'w') as f:
            f.write(codeSub)
            # TODO incomplete
    return src.main.finalize()

def Iterator(iterType : str):
    obj = src.node.IteratorNode(iterType)
    return src.main.addNodeAndLink(None, obj)

def ConcurrentDataBegin(Uin=[], scratch=[]):
    obj = src.node.ConcurrentDataBeginNode(Uin, scratch)
    def linkfn(sourceNode):  # depends on `obj`
        assert sourceNode is not None
        return src.main.addNodeAndLink(sourceNode, obj)
    return linkfn

def ConcurrentDataEnd(Uout, **kwargs):
    obj = src.node.ConcurrentDataEndNode(Uout)
    def linkfn(sourceNode):  # depends on `obj`
        assert sourceNode is not None
        return src.main.addNodeAndLink(sourceNode, obj)
    return linkfn

def Action(name : str, args=list()):
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
