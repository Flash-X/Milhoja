from cgkit.ctree.srctree import SourceTree
import cgkit.ctree.srctree as srctree
import pathlib
import json_sections as jsc
import os
import sys
import re

_SOURCETREE_OPTIONS = {
    'codePath': pathlib.Path.cwd(),
    'indentSpace': ' '*4,
    'verbose': False,
    'verbosePre': '/* ',
    'verbosePost': ' */',
}

####################
# Recipes
####################

def _construct_source_tree(stree: SourceTree, tpl_1: str, helper_path, outer_path):
    """
    Constructs the source tree for the data packet.
    
    :param SourceTree stree: The empty source tree object to use.
    :param str tpl_1: The source template to use.
    :param dict data: The dictionary containing DataPacket JSON data.
    :rtype: None
    """
    stree.initTree(outer_path)
    stree.pushLink( srctree.search_links( stree.getTree() ) )

    # link initial template
    tree_l1  = srctree.load(tpl_1)
    pathInfo = stree.link(tree_l1, linkPath=srctree.LINK_PATH_FROM_STACK)
    if pathInfo:
        stree.pushLink( srctree.search_links(tree_l1) )
    else:
        raise RuntimeError('Linking layer 1 unsuccessful!')
    
    tree_l2  = srctree.load(helper_path)
    pathInfo = stree.link(tree_l2, linkPath=srctree.LINK_PATH_FROM_STACK)
    # if pathInfo is not None and pathInfo:
    # else:
    #     stree.pushLink( srctree.search_links(tree_l2) )
    #     raise RuntimeError('Linking layer 2 unsuccessful!')


def generate_file(data, template_name, output):
    """Generates a source file given a template and output name"""
    template = f'{sys.path[0]}/templates/{template_name}'
    stree = SourceTree(**_SOURCETREE_OPTIONS, debug=False)
    _construct_source_tree(stree, template, data)
    lines = stree.parse()
    if os.path.isfile(output):
        # use logger here but for now just print a warning.
        print(f"Warning: {output} already exists. Overwriting.")
    with open(output, 'w') as new_file:
        lines = re.sub(r'#if 0.*?#endif\n\n', '', lines, flags=re.DOTALL)
        new_file.write(lines)


####################
# Main
####################

def generate_packet_code(data):
    """
    Driver function for constructing the data packet source tree.
    
    :param dict data: The dictionary containing the DataPacket JSON.
    """
    # file_names_all = [(f'{sys.path[0]}/templates/cg-tpl.datapacket_header.cpp', f'{data.get(jsc.NAME, "")}.h'), 
    #                     (f'{sys.path[0]}/templates/cg-tpl.datapacket.cpp', f'{data.get(jsc.NAME, "")}.cpp')]
    # for src,dest in file_names_all:
    #     # assemble from recipe
    #     stree = SourceTree(**_SOURCETREE_OPTIONS, debug=False)
    #     _construct_source_tree(stree, src, data)
    #     # check result
    #     lines = stree.parse()
    #     if os.path.isfile(dest):
    #         # use logger here but for now just print a warning.
    #         print(f"Warning: {dest} already exists. Overwriting.")
    #     with open(dest, 'w') as header:
    #         # Remove garbage from file between #if 0 and #endif to reduce file size.
    #         lines = re.sub(r'#if 0.*?#endif\n\n', '', lines, flags=re.DOTALL)
    #         header.write(lines)
    # print("Assembled datapacket")


if __name__ == '__main__':
    generate_packet_code(None)
