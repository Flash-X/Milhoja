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
    def __init__(self, codePath=None, linkConnectorKeys=['setup', 'execute'],
                 indentSpace=' '*4, verbose=True, debug=False):
        self.linkConnectorKeys = linkConnectorKeys
        self.indentSpace       = indentSpace
        self.verbose           = verbose
        self.debug             = debug
        if codePath is not None:
            self.codePath = codePath
        else:
            self.codePath = pathlib.Path('.')

    def copy(self):
        ''' TODO '''
        return SourceTree(codePath=self.codePath, linkConnectorKeys=self.linkConnectorKeys,
                          indentSpace=self.indentSpace, verbose=self.verbose, debug=self.debug)

    def initialize(self, filename):
        ''' TODO '''
        path = self.codePath / filename
        self.tree         = load(path)
        self.linkLocation = tuple()

    def link(self, treeLink, linkLocation=tuple()):
        ''' TODO '''
        assert isinstance(linkLocation, tuple), type(linkLocation)
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
            raise TypeError('Expected dict or path, but got {}'.format(type(treeLink)))
        # perform linking
        return SourceTree.link_trees(c, l, self.linkConnectorKeys)

    @staticmethod
    def link_trees(tree, treeLink, linkConnectorKeys):
        assert isinstance(tree, dict), type(tree)
        assert isinstance(treeLink, dict), type(treeLink)
        topLevelParams = dict()
        SourceTree._gather_parameters(treeLink, topLevelParams)
        locations = list()
        for label in linkConnectorKeys:
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
        ''' TODO '''
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
        lines  = list()
        params = dict()
        SourceTree.parse_tree(self.tree, lines, params, self.indentSpace, 0, self.verbose, self.debug)
        # parse lines
        return '\n'.join(lines)

    @staticmethod
    def parse_tree(tree, lines, params, indentSpace, indentCount, verbose, debug):
        assert isinstance(tree, dict), type(tree)
        assert isinstance(lines, list), type(lines)
        assert isinstance(params, dict), type(params)
        # gather parameters
        indentCount += SourceTree._gather_parameters(tree, params)
        indent       = indentSpace * indentCount
        # process tree
        hasConn = False
        hasCode = False
        hasLink = False
        for key in tree.keys():
            if '_connector' in key:
                hasConn = True
                # skip over connector
                if verbose:
                    lines.append(indent + SourceTree._verbose_begin(key))
                SourceTree.parse_tree(tree[key], lines, copy.deepcopy(params), indentSpace, indentCount, verbose, debug)
                if verbose:
                    lines.append(indent + SourceTree._verbose_end(key))
            elif '_code' in key:
                hasCode = True
                # process code
                if verbose and '_param:__file__' in tree:
                    lines.append(indent + SourceTree._verbose_begin(tree['_param:__file__']))
                for c in tree['_code']:
                    if isinstance(c, str):
                        lines.append(indent + SourceTree._substitute_parameters(c, params))
                    elif isinstance(c, dict):
                        SourceTree.parse_tree(c, lines, copy.deepcopy(params), indentSpace, indentCount, verbose, debug)
                    else:
                        raise TypeError('Expected str or dict, but got {}'.format(type(c)))
                if verbose and '_param:__file__' in tree:
                    lines.append(indent + SourceTree._verbose_end(tree['_param:__file__']))
            elif '_link' in key:
                hasLink = True
                # recurse into links
                if verbose:
                    lines.append(indent + SourceTree._verbose_begin(key))
                for lt in tree[key]:
                    SourceTree.parse_tree(lt, lines, copy.deepcopy(params), indentSpace, indentCount, verbose, debug)
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
