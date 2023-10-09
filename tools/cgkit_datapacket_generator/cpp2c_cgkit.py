from cgkit.ctree.srctree import SourceTree
import cgkit.ctree.srctree as srctree
import pathlib
import json_sections as jsections
import os
import sys

_SOURCETREE_OPTIONS = {
    'codePath': pathlib.Path('.'),
    'indentSpace': ' '*4,
    'verbose': False,
    'verbosePre': '/* ',
    'verbosePost': ' */',
}

_OUTPUT = 'cgkit.cpp2c.cxx'

####################
# Recipes
####################

def _construct_source_tree(stree: SourceTree, tpl_1: str, data: dict):
    """
    Constructs the source tree for the cpp to c layer.
    
    :param SourceTree stree: The source tree
    :param str tpl_1: The file path for the initial template.
    :param dict data: The data containing the data packet JSON information.
    """
    init = 'cg-tpl.cpp2c_outer.cpp'
    helpers = 'cg-tpl.cpp2c_helper.cpp'
    extra_queue = f'{sys.path[0]}/templates/'
    if data[jsections.EXTRA_STREAMS] == 0:
        extra_queue = f'{extra_queue}cg-tpl.cpp2c_no_extra_queue.cpp'
    else:
        extra_queue = f'{extra_queue}cg-tpl.cpp2c_extra_queue.cpp'

    # load outer template
    stree.initTree(init)
    stree.pushLink( srctree.search_links(stree.getTree()) )

    # load and link each template into source tree.
    for idx,link in enumerate([tpl_1, helpers, extra_queue]):
        tree_link = srctree.load(link)
        pathInfo = stree.link(tree_link, linkPath=srctree.LINK_PATH_FROM_STACK)
        if pathInfo:
            stree.pushLink(srctree.search_links(tree_link))
        else:
            raise RuntimeError(f'Linking layer {idx} ({link}) unsuccessful!')

####################
# Main
####################

def generate_datapacket_cpp2c_layer(data):
    # assemble from recipe
    stree = SourceTree(**_SOURCETREE_OPTIONS, debug=False)
    _construct_source_tree(stree, f'{sys.path[0]}/templates/cg-tpl.cpp2c.cpp', data)
    # check result
    output = f'{data["name"]}.cpp2c.cxx'
    lines = stree.parse()
    if os.path.isfile(output):
        print(f'Warning: {output} already exists. Overwriting.')
    with open(output, 'w') as cpp2c:
        cpp2c.write(lines)
    print("Assembled cpp2c layer.")

if __name__ == '__main__':
    generate_datapacket_cpp2c_layer(None)