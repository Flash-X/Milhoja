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

OUTER = 'cg-tpl.datapacket_outer.cpp'
HELPERS = "cg-tpl.datapacket_helpers.cpp"

####################
# Recipes
####################

def constructSourceTree(stree, tpl_1, data: dict):
    init = OUTER if not data else data["outer"]
    helpers = HELPERS if not data else data["helpers"]
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
    # if pathInfo is not None and pathInfo:
    # else:
    #     stree.pushLink( srctree.search_links(tree_l2) )
    #     raise RuntimeError('Linking layer 2 unsuccessful!')

####################
# Main
####################

def main(data):
    file_names_all = [('cg-tpl.datapacket_header.cpp', 'cgkit.datapacket.h'), 
                        ('cg-tpl.datapacket.cpp', 'cgkit.datapacket.cpp')]
    for src,dest in file_names_all:
        # assemble from recipe
        stree = SourceTree(**SOURCETREE_OPTIONS, debug=False)
        constructSourceTree(stree, src, data)
        # check result
        lines = stree.parse()
        with open(dest, 'w') as header:
            header.write(lines)
    print("Assembled datapacket")

if __name__ == '__main__':
    main(None)
