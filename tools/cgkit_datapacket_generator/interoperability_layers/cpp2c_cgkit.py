from cgkit.ctree.srctree import SourceTree
import cgkit.ctree.srctree as srctree
import pathlib

SOURCETREE_OPTIONS = {
    'codePath': pathlib.Path('.'),
    'indentSpace': ' '*2,
    'verbose': False,
    'verbosePre': '/* ',
    'verbosePost': ' */',
}

OUTPUT = 'cgkit.dr_hydroAdvance_bundle.cxx'

####################
# Recipes
####################

def constructSourceTree(stree, tpl_1, data: dict):
    init = 'cg-tpl.cpp2c_outer.cpp'
    helpers = 'cg-tpl.cpp2c_helper.cpp'

    stree.initTree(init)
    stree.pushLink( srctree.search_links(stree.getTree()) )
    tree_l1  = srctree.load(tpl_1)
    pathInfo = stree.link(tree_l1, linkPath=srctree.LINK_PATH_FROM_STACK)
    if pathInfo is not None and pathInfo:
        stree.pushLink( srctree.search_links(tree_l1) )
    else:
        raise RuntimeError('Linking layer 1 unsuccessful!')
    tree_l2  = srctree.load(helpers)
    pathInfo = stree.link(tree_l2, linkPath=srctree.LINK_PATH_FROM_STACK)

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
