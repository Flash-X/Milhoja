import src.main
import src.node

def initializeCodeGenerator(codeAssembler):
    return src.main.initialize(codeAssembler)

def finalizeCodeGenerator(basename=None):
    # perform setup and parse code
    src.main.setUp()
    driverCode, subroutineCode = src.main.parseCode()
    if basename is not None:
        # write driver code
        if driverCode is not None:
            with open('_code_{}_driver.cpp'.format(basename), 'w') as f:
                f.write(driverCode)
        # write subroutine code
        if subroutineCode is not None:
            for nameSub, codeSub in subroutineCode.items():
                with open('_code_{}_{}.cpp'.format(basename, nameSub), 'w') as f:
                    f.write(codeSub)
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
