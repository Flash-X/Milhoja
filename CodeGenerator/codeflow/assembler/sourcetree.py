import copy, json, pathlib, re


def _load_from_json(path):
    assert isinstance(path, pathlib.Path), type(path)
    with open(path, 'r') as f:
        return json.load(f)
    return dict()


def _load_from_source(path):
    assert isinstance(path, pathlib.Path), type(path)
    lines = path.read_text().splitlines()
    tree  = dict()
    # split up sections of source code
    split_indices = list()
    for i, line in enumerate(lines):
        if '_connector' in line:
            split_indices.append(i)
    split_indices.append(len(lines))
    if 0 != split_indices[0]:
        split_indices = [0] + split_indices
    # generate tree from source code
    for split_idx in range(len(split_indices) - 1):
        lines_split = lines[split_indices[split_idx]:split_indices[split_idx+1]]
        lineno_remove = list()
        if '_connector' in lines_split[0]:
            connector = re.search(r'\b_connector[:\w]*\b', lines_split[0])
            if connector:
                connector = connector.group()
                lineno_remove.append(0)
        else:
            connector = None
        indent = 0
        params = dict()
        for i, line in enumerate(lines_split):
            if '{' in line:
                indent += 1
            if '}' in line:
                indent -= 1
            #if '_param:' in line:
            if re.search(r'[/\*]{2}\s*_param', line):
                matches = re.findall(r'\b_param:\w+\b[=\s]+\b\w+\b', line)
                for m in matches:
                    s = re.split(r'[=\s]+', m)
                    assert 2 == len(s)
                    params[s[0]] = s[1]
                if matches:
                    lineno_remove.append(i)
            if '_link' in line:
                linkName = re.search(r'\b_link[:\w]*\b', line)
                if linkName:
                    linkName = linkName.group()
                    lines_split[i] = { '_param:indent': indent, linkName: [] }
        # remove lines
        lineno_remove.reverse()
        for i in lineno_remove:
            lines_split.pop(i)
        # add items to tree
        if connector:
            assert isinstance(connector, str), type(connector)
            tree[connector] = {'_code': lines_split}
            tree[connector].update(params)
        else:
            tree.update({'_code': lines_split})
            tree.update(params)
    # return tree
    return tree


def load(path):
    ''' TODO '''
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    fileType = path.suffix
    if fileType.lower() == '.json':  # if load JSON
        tree = _load_from_json(path)
    elif fileType.lower() in ['.c', '.cc', '.cpp']:  # if load C/C++ source
        tree = _load_from_source(path)
    else:  # otherwise file is unknown
        raise NotImplementedError('File type "{}" is not supported'.format(fileType))
    # remove connectors from top layer
    #TODO there is probably no need for this top layer stuff
#   if isTopLayer:
#       items = dict()
#       for treekey in tree.keys():
#           if '_connector' in treekey.lower():
#               popped = tree.pop(treekey, None)
#               for key, value in popped.items():
#                   if '_code' == key:
#                       try:
#                           items[key].extend(value)
#                       except KeyError:
#                           items[key] = value
#                       except:
#                           raise
#                   else:
#                       items[key] = value
#       tree.update(items)
    # add file name as parameter
    tree['_param:__file__'] = path.name
    return tree


def dump(tree, path, indent=2):
    ''' TODO '''
    assert isinstance(tree, dict), type(tree)
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    fileType = path.suffix
    with open(path, 'w') as f:
        if '.json' == fileType.lower():
            json.dump(tree, f, indent=indent)
        else:
            raise NotImplementedError('File type "{}" is not supported'.format(fileType))


class SourceTree():
    def __init__(self, codePath=None, indentSpace=' '*4, verbose=True, debug=False):
        if codePath is not None:
            self.codePath = codePath
        else:
            self.codePath = pathlib.Path('.')
        self.indentSpace = indentSpace
        self.verbose     = verbose
        self.debug       = debug
        self.tree        = None

    def copy(self):
        ''' TODO '''
        return SourceTree(codePath=self.codePath, indentSpace=self.indentSpace,
                          verbose=self.verbose, debug=self.debug)

    def initialize(self, filename, parameters=dict()):
        ''' TODO '''
        path = self.codePath / filename
        self.tree         = load(path)
        for key, value in parameters.items():
            assert '_param' in key
            self.tree[key] = value

    def dump(self, path, indent=2):
        assert self.tree, 'Tree does not exist, call initialize() first.'
        return dump(self.tree, path, indent)

    def link(self, treeLink, linkLocation=tuple(), parameters=dict()):
        ''' TODO '''
        assert isinstance(linkLocation, tuple), type(linkLocation)
        assert self.tree, 'Tree does not exist, call initialize() first.'
        # set connector
        c = self.tree
        for loc in linkLocation:
            c = c[loc]
        # set link
        if isinstance(treeLink, dict):
            l = treeLink
        elif isinstance(treeLink, str) or isinstance(treeLink, pathlib.Path):
            l = load(treeLink)
        else:
            raise TypeError('Expected dict or filepath, but got {}'.format(treeLink))
        # perform linking
        locations = SourceTree.link_trees(c, l, copy.deepcopy(parameters))
        for i, loc in enumerate(locations):
            locations[i] = tuple(list(linkLocation) + list(loc))
        return locations

    @staticmethod
    def link_trees(tree, treeLink, parameters=dict()):
        assert isinstance(tree, dict), type(tree)
        assert isinstance(treeLink, dict), type(treeLink)
        assert isinstance(parameters, dict), type(parameters)
        # get top-level parameters of to-be-linked tree
        SourceTree._gather_parameters(treeLink, parameters)
        # link trees
        linkLocations = list()
        for connKey in SourceTree._gather_connectors(treeLink):
            # set link key
            s = re.split(r'[:]{1}', connKey)
            if 1 == len(s):
                linkKey = '_link'
            elif 2 == len(s):
                linkKey = '_link:' + s[1]
            else:
                raise NotImplementedError('Connector key "{}" is not supported'.format(connKey))
            # init connector dict
            c = copy.deepcopy(treeLink[connKey])
            for key, value in parameters.items():
                assert '_param' in key
                if not key in c:
                    c[key] = value
            # attach connector to link
            if '_code' in tree:
                for i in range(len(tree['_code'])):  # loop over all code lines
                    if isinstance(tree['_code'][i], dict) and (linkKey in tree['_code'][i]):
                        tree['_code'][i][linkKey].append(c)
                        j = len(tree['_code'][i][linkKey]) - 1
                        linkLocations.append(('_code', i, linkKey, j))
        return linkLocations

    def parse(self):
        ''' TODO '''
        assert self.tree, 'Tree does not exist, call initialize() first.'
        # write debug file
        if self.debug:
            try:
                cl = self.__class__.__name__
                fn = self.tree['_param:__file__']
                name = '{}_{}'.format(cl, fn)
            except KeyError:
                cl = self.__class__.__name__
                name = '{}'.format(cl)
            except:
                raise
            dump(self.tree, '_debug_{}.json'.format(name))
        # parse tree
        lines      = list()
        parameters = dict()
        SourceTree.parse_tree(self.tree, lines, parameters, self.indentSpace, 0, self.verbose, self.debug)
        # parse lines
        return '\n'.join(lines)

    @staticmethod
    def parse_tree(tree, lines, parameters, indentSpace, indentCount, verbose, debug):
        assert isinstance(tree, dict), type(tree)
        assert isinstance(lines, list), type(lines)
        assert isinstance(parameters, dict), type(parameters)
        # gather parameters
        indentCount += SourceTree._gather_parameters(tree, parameters)
        indent       = indentSpace * indentCount
        try:
            whichFile = tree['_param:__file__']
        except KeyError:
            whichFile = None
        # process tree
        hasConn = False
        hasCode = False
        hasLink = False
        for key in tree.keys():
            if '_connector' in key:
                hasConn = True
                # skip over connector
                if verbose:
                    if whichFile:
                        lines.append(indent + SourceTree._verbose_begin(key+' file='+whichFile))
                    else:
                        lines.append(indent + SourceTree._verbose_begin(key))
                SourceTree.parse_tree(tree[key], lines, copy.deepcopy(parameters), indentSpace, indentCount, verbose, debug)
                if verbose:
                    lines.append(indent + SourceTree._verbose_end(key))
            elif '_code' in key:
                hasCode = True
                # process code
                if verbose and whichFile:
                    lines.append(indent + SourceTree._verbose_begin(whichFile))
                for c in tree['_code']:
                    if isinstance(c, str):
                        lines.append(indent + SourceTree._substitute_parameters(c, parameters))
                    elif isinstance(c, dict):
                        SourceTree.parse_tree(c, lines, copy.deepcopy(parameters), indentSpace, indentCount, verbose, debug)
                    else:
                        raise TypeError('Expected str or dict, but got {}'.format(type(c)))
                if verbose and whichFile:
                    lines.append(indent + SourceTree._verbose_end(whichFile))
            elif '_link' in key:
                hasLink = True
                # recurse into links
                if verbose:
                    lines.append(indent + SourceTree._verbose_begin(key))
                for lt in tree[key]:
                    SourceTree.parse_tree(lt, lines, copy.deepcopy(parameters), indentSpace, indentCount, verbose, debug)
                if verbose:
                    lines.append(indent + SourceTree._verbose_end(key))
            else:
                if not ('_param' in key):
                    raise SyntaxError('Unknown key: {}'.format(key))
        assert 1 == sum([hasConn, hasCode, hasLink]), (hasConn, hasCode, hasLink)

    @staticmethod
    def _verbose_begin(tag):
        return r'/* <' + tag + '> */'

    @staticmethod
    def _verbose_end(tag):
        return r'/* </' + tag + '> */'

    @staticmethod
    def _gather_connectors(tree):
        connectors=list()
        for key in tree.keys():
            if key.startswith('_connector'):
                connectors.append(key)
        return connectors

    @staticmethod
    def _gather_parameters(tree, parameters):
        indentCount = 0
        for key, value in tree.items():
            if key.startswith('_param:'):
                if '_param:indent' == key:
                    indentCount += value
                else:
                    parameters[key] = value
        return indentCount

    @staticmethod
    def _substitute_parameters(line, parameters):
        assert isinstance(line, str), type(line)
        assert isinstance(parameters, dict), type(parameters)
        if not ('_param:' in line):
            return line
        for key, value in parameters.items():
            line = re.sub(r'\b'+str(key)+r'\b', str(value), line)
        return line
