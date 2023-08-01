from cgkit.ctree.srctree import SourceTree
import cgkit.ctree.srctree as srctree
import pathlib
import json_sections as jsections

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

def constructSourceTree(stree: SourceTree, tpl_1: str, data: dict):
    """
    Constructs the source tree for the cpp to c layer.
    
    :param SourceTree stree: The source tree
    :param str tpl_1: The file path for the initial template.
    :param dict data: The data containing the data packet JSON information.
    """
    init = 'cg-tpl.cpp2c_outer.cpp'
    helpers = 'cg-tpl.cpp2c_helper.cpp'
    extra_queue = 'cg-tpl.cpp2c_no_extra_queue.cpp' if data[jsections.EXTRA_STREAMS] == 0 else 'cg-tpl.cpp2c_extra_queue.cpp'

    # load outer template
    stree.initTree(init)
    stree.pushLink( srctree.search_links(stree.getTree()) )

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

def main(data):
    # assemble from recipe
    stree = SourceTree(**_SOURCETREE_OPTIONS, debug=False)
    constructSourceTree(stree, 'cg-tpl.cpp2c.cpp', data)
    # check result
    lines = stree.parse()
    with open(_OUTPUT, 'w') as cpp2c:
        cpp2c.write(lines)
    print("Assembled to cpp2c layer.")

if __name__ == '__main__':
    main(None)
