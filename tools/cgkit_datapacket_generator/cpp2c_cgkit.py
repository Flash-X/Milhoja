from cgkit.ctree.srctree import SourceTree
import cgkit.ctree.srctree as srctree
import pathlib
import json_sections as jsections

SOURCETREE_OPTIONS = {
    'codePath': pathlib.Path('.'),
    'indentSpace': ' '*4,
    'verbose': False,
    'verbosePre': '/* ',
    'verbosePost': ' */',
}

OUTPUT = 'cgkit.cpp2c.cxx'

####################
# Recipes
####################

def constructSourceTree(stree, tpl_1, data: dict):
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
    tree_l1  = srctree.load(tpl_1)
    pathInfo = stree.link(tree_l1, linkPath=srctree.LINK_PATH_FROM_STACK)
    if pathInfo is not None and pathInfo:
        stree.pushLink( srctree.search_links(tree_l1) )
    else:
        raise RuntimeError('Linking layer 1 unsuccessful!')
    
    # Load generated helpers template
    tree_l2  = srctree.load(helpers)
    pathInfo = stree.link(tree_l2, linkPath=srctree.LINK_PATH_FROM_STACK)
    if pathInfo:
        stree.pushLink( srctree.search_links(tree_l2) )
    else:
        raise RuntimeError('Linking layer 2 unsuccessful!')
    
    # load the extra queue information
    tree_l3 = srctree.load(extra_queue)
    pathInfo = stree.link(tree_l3, linkPath=srctree.LINK_PATH_FROM_STACK)
    if pathInfo:
        stree.pushLink( srctree.search_links(tree_l3) )
    else:
        raise RuntimeError('Linking layer 3 unsuccessful!')

####################
# Main
####################

def main(data):
    # assemble from recipe
    stree = SourceTree(**SOURCETREE_OPTIONS, debug=False)
    constructSourceTree(stree, 'cg-tpl.cpp2c.cpp', data)
    # check result
    lines = stree.parse()
    with open(OUTPUT, 'w') as cpp2c:
        cpp2c.write(lines)
    print("Assembled to cpp2c layer.")

if __name__ == '__main__':
    main(None)
