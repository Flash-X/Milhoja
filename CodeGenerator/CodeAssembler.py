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

    def copy(self):
        return CodeAssembler(codePath=self.codePath, indentSpace=self.indentSpace,
                             verbose=self.verbose, debug=self.debug)

    def initializeDriver(self):
####DEV
        path_default = self.codePath / 'ex_sedov_driver_Default.cpp'
####DEV
        self.tree = CodeAssembler.load(path_default, isTopLayer=True)
        self.linkLocation = tuple()

    def initializeSubroutine(self, functionName=None, device=None):
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

    @staticmethod
    def load(path, isTopLayer=False):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        fileType = path.suffix
        if '.json' == fileType.lower():  # if loading JSON file
            with open(path, 'r') as f:
                tree = json.load(f)
            # remove connectors from top layer
            if isTopLayer:
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
        elif '.cpp' == fileType.lower():  # if loading CPP file
            code = path.read_text().splitlines()
            # process parameters and links
            indent = 0
            params = dict()
            for i, line in enumerate(code):
                if '{' in line:
                    indent += 1
                if '}' in line:
                    indent -= 1
                if '_param:' in line:
                    matches = re.findall(r'\b_param:\w+\b[=\s]+\b\w+\b', line)
                    for m in matches:
                        s = re.split(r'[=\s]+', m)
                        assert 2 == len(s)
                        params[s[0]] = s[1]
                if '_link:' in line:
                    linkName = re.search(r'\b_link:\w+\b', line).group()
                    assert linkName in ['_link:setup', '_link:execute']
                    code[i] = { '_param:indent': indent, linkName: [] }
            # assemble code tree
            if isTopLayer:
                tree = { '_code': code }
            else:
                tree = { '_connector:execute': {'_code': code} }
            tree.update(params)
        else:  # otherwise file is unknown
            raise NotImplementedError('File type "{}" is not supported'.format(fileType))
        # add file name as parameter
        tree['_param:__file__'] = path.name
        return tree

    @staticmethod
    def dump(tree, path):
        assert isinstance(tree, dict), type(tree)
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        fileType = path.suffix
        with open(path, 'w') as f:
            if '.json' == fileType:
                json.dump(tree, f, indent=4)
            else:
                raise NotImplementedError('File type "{}" is not supported'.format(fileType))

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
            try:
                fn = '{}_{}'.format(self.tree['_param:functionName'], self.device)
            except (AttributeError, KeyError):
                fn = 'DEV'
                #TODO set proper function name
            except:
                raise
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
