import re

from pathlib import Path
from . import LogicError, AbcLogger
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
    output: Path, linked_templates: list, overwrite: bool, logger: AbcLogger,
    sourcetree_opts=DEFAULT_SOURCE_TREE_OPTS
):
    """
    Generates a DataPacket file based on the inputs with CG-Kit.

    :param str output: The output name for the file.
    :param list linked_templates: All templates to be linked. The first in the
                                  list is the initial template. Keep in mind
                                  that this means that order matters when
                                  passing in templates to link!
    :param bool overwrite: Whether or not to overwrite any existing files.
    :param logger: The logger to use for output.
    :param dict sourcetree_opts: The list of options used to generate files.
                                 Default is set to DEFAULT_SOURCE_TREE_OPTS,
                                 but it's possible for other code generators
                                 to pass in their own options.
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
    if lines[-1] != "\n":
        lines = lines + "\n"

    if output.is_file():
        logger.warn(caller, f"{str(output)} already exists.")
        if not overwrite:
            logger.error(caller, "Overwrite is set to False.")
            raise FileExistsError("File already exists.")

    with open(output, 'w') as new_file:
        # remove anything between #if 0 directive to reduce file sizes.
        # There may be extra garbage code inside of a file if links are shared
        #  between templates.
        lines = re.sub(r'#if 0.*?#endif\n\n', '', lines, flags=re.DOTALL)
        # Write code to the destination
        new_file.write(lines)
        # end of file
        new_file.write("\n")
