import re

from pathlib import Path
from . import LogicError
from . import AbcLogger
from cgkit.ctree import srctree
from cgkit.ctree.srctree import SourceTree

DEFAULT_SOURCE_TREE_OPTS = {
    'codePath': Path.cwd(),
    'indentSpace': ' '*4,
    'verbose': False,
    'verbosePre': '/* ',
    'verbosePost': ' */',
}


def generate_packet_file(
    output: Path,
    linked_templates: list,
    overwrite: bool,
    logger: AbcLogger,
    sourcetree_opts=DEFAULT_SOURCE_TREE_OPTS,
):
    """
    Generates a data packet file for creating the entire packet.

    :param str output: The output name for the file.
    :param dict sourcetree_opts: The dictionary containing the
                                sourcetree parameters. Default is provided
                                but code generators can pass their own opts
                                in.
    :param list linked_templates: All templates to be linked.
                                The first in the list is
                                the initial template.
    """
    caller = "Milhoja generate_packet_file"

    def construct_source_tree(stree: SourceTree, templates: list):
        assert len(templates) > 0
        stree.initTree(templates[0])
        stree.pushLink(srctree.search_links(stree.getTree()))

        # load and link each template into source tree.
        for link in templates[1:]:
            tree_link = srctree.load(link)
            pathInfo = stree.link(
                tree_link, linkPath=srctree.LINK_PATH_FROM_STACK
            )
            if pathInfo:
                stree.pushLink(srctree.search_links(tree_link))
            else:
                logger.error(caller, "Linking unsuccessful!")
                raise LogicError("Link unsuccessful.")

    # Generates a source file given a template and output name
    stree = SourceTree(**sourcetree_opts, debug=False)
    construct_source_tree(stree, linked_templates)
    lines = stree.parse()

    if output.is_file():
        logger.warn(caller, f"{str(output)} already exists.")
        if not overwrite:
            logger.error(caller, "Overwrite is set to False.")
            raise FileExistsError("File already exists.")

    with open(output, 'w') as new_file:
        lines = re.sub(r'#if 0.*?#endif\n\n', '', lines, flags=re.DOTALL)
        new_file.write(lines)
