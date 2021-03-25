import copy, json, pathlib, re
import pprint
pp = pprint.PrettyPrinter(indent=4)

path = pathlib.Path('../example')
path_generic = path / 'ex_sedov_subroutine_generic.json'
path_cpu     = path / 'ex_sedov_subroutine_cpu.json'

with open(path_generic, 'r') as f:
    dict_generic = json.load(f)
with open(path_cpu, 'r') as f:
    dict_cpu = json.load(f)

#pp.pprint(dict_generic)
#pp.pprint(dict_cpu)

###############################################################################

def merge_dicts(d, dmerge):
    assert isinstance(d, dict), type(d)
    assert isinstance(dmerge, dict), type(dmerge)
    # merge parameters
    for key, value in dmerge.items():
        if key.startswith('_param:'):
            d[key] = value
    # merge "setup" code
    if '_code:setup' in dmerge:
        code = copy.deepcopy(dmerge['_code:setup'])
        try:
            d['_code:setup'].extend(code)
        except KeyError:
            d['_code:setup'] = code
    # merge "execute" code
    if '_code:execute' in dmerge:
        code = copy.deepcopy(dmerge['_code:execute'])
        try:
            d['_code:execute'].extend(code)
        except KeyError:
            d['_code:execute'] = code
    return d

def link_dicts(d, dlink):
    assert isinstance(d, dict), type(d)
    assert isinstance(dlink, dict), type(dlink)
    # process "execute" code
    if '_connector:execute' in dlink:
        try:
            for c in d['_code:execute']:
                if isinstance(c, dict):
                    c['_link:execute'].append(copy.deepcopy(dlink['_connector:execute']))
        except KeyError:
            pass
    return d

def _substitute_params(line, params):
    assert isinstance(line, str), type(line)
    assert isinstance(params, dict), type(params)
    if not ('_param:' in line):
        return line
    for key, value in params.items():
        line = re.sub(r'\b'+str(key)+r'\b', str(value), line)
    return line

def process_dict(d, lines=list(), params=dict(), indent=4, verbose=True):
    assert isinstance(d, dict), type(d)
    assert isinstance(lines, list), type(lines)
    assert isinstance(params, dict), type(params)
    # gather parameters
    if not params:
        params['_param:indent'] = 0
    for key, value in d.items():
        if key.startswith('_param:'):
            if '_param:indent' == key:
                params[key] += value
            else:
                params[key] = value
    # set indent space
    indentSpace = ' ' * indent * params['_param:indent']
    # initialize code blocks
    if not ('_code:setup' in d):
        d['_code:setup'] = list()
    if not ('_code:execute' in d):
        d['_code:execute'] = list()
    # process "setup" link
    try:
        for ld in d['_link:setup']:
            d['_code:setup'].extend(ld['_code:setup'])
            assert not ('_code:execute' in ld)
    except KeyError:
        pass
    # process "execute" link
    try:
        for ld in d['_link:execute']:
            d['_code:setup'].extend(ld['_code:setup'])
            d['_code:execute'].extend(ld['_code:execute'])
    except KeyError:
        pass
    # process "setup" code
    if verbose:
        lines.append(indentSpace + r'/* <_code:setup> */')
    for c in d['_code:setup']:
        if isinstance(c, str):
            lines.append(indentSpace + _substitute_params(c, params))
        elif isinstance(c, dict):
            process_dict(c, lines, params)
        else:
            raise TypeError('Expected str or dict, but got {}'.format(type(c)))
    if verbose:
        lines.append(indentSpace + r'/* </_code:setup> */')
    # process "execute" code
    if verbose:
        lines.append(indentSpace + r'/* <_code:execute> */')
    for c in d['_code:execute']:
        if isinstance(c, str):
            lines.append(indentSpace + _substitute_params(c, params))
        elif isinstance(c, dict):
            process_dict(c, lines, params)
        else:
            raise TypeError('Expected str or dict, but got {}'.format(type(c)))
    if verbose:
        lines.append(indentSpace + r'/* </_code:execute> */')
    return lines

def process_lines(lines):
    return '\n'.join(lines)

d     = merge_dicts(dict_generic, dict_cpu)
d     = link_dicts(d, dict_cpu)
lines = process_dict(d)
code  = process_lines(lines)
#pp.pprint(lines)
print('---', code, '---', sep='\n')
