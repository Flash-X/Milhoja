import re
from warnings import warn

from sys import maxsize

from . import (
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT, TILE_LBOUND_ARGUMENT,
    TILE_UBOUND_ARGUMENT, LogicError
)


class IncorrectFormatException(BaseException):
    """Thrown when the format of lbound or extents is incorrect."""
    pass


class NonIntegerException(BaseException):
    """Thrown when an extents or lbound contains a float value."""
    pass


def parse_lbound(lbound: str) -> list:
    """
    Parses an lbound string for use within the generator.

    This function is deprecated and only exists for the C++ tests.
    Once C++ code generation is updated to use parse_lbound_f, this function
    will be removed.

    ..todo::
        * This lbound parser only allows simple lbounds.
          The current format does not allow nested arithmetic expressions
          or more than 2 intvects being combined. If we wanted complex
          mathematics we would need to incorporate an actual math parser.
        * Write more concrete tests for lbound parsing.
        * What are the restrictions on an lbound string? Because of the
          way that FArray4D works in I'm assuming that tile_ can only
          appear in the first or last index of the lbound list. Is this a
          valid assumption? I'm not sure if there's much of a choice with
          the way lbound works. There's no way to assume that tile_ cam
          appear anywhere
        * I can improve the current iteration of this function by splitting
          any inserted tile metadata bounds into its components, then
          iterating over all segments of the lbound and stitching together
          the components in each space. After, I can recombine the lbound
          back into its original form and save any integers being added to it
          and converting it into an IntVect. But I don't have time to do this
          right now so I'm just going to clean lbound as is.

    :param str lbound: The lbound string to parse.
    """
    warn(
        'parse_lbound is deprecated. Use parse_lbound_f instead.',
        DeprecationWarning, stacklevel=2
    )

    keywords = {
        TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
        TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT
    }
    # find all words in teh lbound string and ensure that they contain valid
    # keywords.
    words = re.findall(r'\b(?:[\w]+)\b', lbound)
    # just use python to throw out all numeric values because I'm bad at
    # regular expressions.
    words = [word for word in words if not word.isnumeric()]
    for word in words:
        if word not in keywords:
            raise NotImplementedError(
                f"{lbound} contained word not in {keywords}"
            )

    # remove tile_ prefix from keywords
    lbound = lbound.replace("tile_", '').replace(' ', '')
    # find everything between a single set of parens.
    regexr = r'\(([^\)]+)\)'
    matches = re.findall(regexr, lbound)
    stitch = ''

    # find stitching arithmetic
    # ..todo::
    #    * allow math expressions inside intvect constructors?
    if len(matches) > 1:
        if len(matches) > 2:
            raise NotImplementedError(
                "Complex lbound expressions are not implemented yet!"
            )
        find_math_string = lbound
        for match in matches:
            find_math_string = find_math_string.replace(f"({match})", '')
        symbols = re.findall(r'[\+\-\/\*]', find_math_string)
        if len(symbols) > 1:
            raise IncorrectFormatException(
                "lbound only supports one math op symbol for now."
            )
        stitch = symbols[0]

    lbound_parts = []
    for group in matches:
        group = group.split(',')
        assert len(group) > 0

        # check the size of the array. If it's > 4 then it's not a valid
        # lbound FOR NOW**. We need actual test cases for arrays > 4.
        size = 0
        for item in group:
            amount = 3 if "tile_" + item in keywords else 1
            size += amount
            if size > 4:
                raise NotImplementedError("The size of lbound is too large.")

        if len(group) == 1:
            lbound_parts.append((f'({group[0]})', None))
        # we have a keyword in slot 0
        elif "tile_" + group[0] in keywords:
            ncomp = group[1] if len(group) > 1 else None
            lbound_parts.append((f'({group[0]})', ncomp))
        # easy case, all values are integers
        # check if there are any alphabetical chars inside of the string.
        elif all([not re.search(r'[a-zA-z]', value) for value in group]):
            # We use this vector in case the scratch array dim < 3D.
            # All FArrayND constructors take an intvect as the first param.
            # reminder that all int vects are length 3.
            init_vect = ['1', '1', '1']
            ncomp = None
            for idx, value in enumerate(init_vect):
                init_vect[idx] = str(group[idx])
            if len(group) > len(init_vect):
                # there should never be a case where its greater than 4, since
                # FArray > 4D does not exist anyway.
                assert len(group) == 4
                ncomp = group[-1]
            # join together wrapped in an int vect
            lbound_parts.append(
                (f'IntVect{{LIST_NDIM({",".join(init_vect)})}}', ncomp)
            )
        else:
            raise NotImplementedError(
                f"This lbound pattern has not been implemented yet: {lbound}"
            )

    # stitch the lbound parts together with arithmetic symbol
    results = []
    for item in lbound_parts:
        if item[0]:
            if len(results) == 0:
                results.append(item[0])
            else:
                results[0] = results[0] + stitch + item[0]

            if item[1]:
                if len(results) == 1:
                    results.append(item[1])
                else:
                    results[1] = results[1] + stitch + item[1]
    results = [item for item in results if item]
    return results


def parse_lbound_f(lbound: str):
    """
    Parses a given lbound string and returns a list containing all parts of
    the lbound for use in packing information.

    Formats for lbounds should be limited to a parenthesis enclosed string
    with comma separation between elements. The lbound is allowed to contain
    tile_lo, tile_hi, tile_lbound, and tile_ubound. Each lbound string is also
    allowed to use non-nested mathematic expressions between parenthesis.
    Any available keyword to be used inside of an lbound string is considered
    to be a "size 3" insertion. Since mathematic expressions that use lbounds
    are required to have all lists be the same size, it is important to
    understand how large the lbound string is.

    Examples of valid formats include:
        * (1, 2, -3, 4)
        * (tile_lo, 1)
        * (1, tile_lbound),
        * (tile_lo) - (1, 1, 1)
        * (tile_lbound, 1) + (1, 3, 4, 5)
        * (tile_lo, tile_lo) - (tile_lbound, tile_lbound)
        * (1, 2, 3) + (4, 5, 6) - (2, 2, 2) * (1, 2, 3)

    Examples of invalid lbound formats:
        * (2, (3-4-6), 2, (92))
        * (tile_lo, tile_lo) - (tile_lo)
        * (1, 2, 3) + (tile_lo, 2, 3, 4)

    todo::
        * We have to force the calling code to replace any variables
          that are used in the lbound with their source name, and then replace
          the source names with the original variable name when the list is
          returned.
        * This parser does not handle scalar addition/mult/sub/div with
          metadata. and does not yet throw an error when encountering it.
        * This parser does not support nested expressions due to the
          limitations of regular expressions. A full lbound parser would use a
          tokenizer to extract the full expression.
        * Write a clear set of rules in the docs of what a valid lbound is.

    :param str lbound: The lbound string to parse.
    :return: A tuple containing the list of the parsed expression and the list
             of keywords found within the expression.
    """
    keywords = {
        TILE_LO_ARGUMENT, TILE_HI_ARGUMENT, TILE_LBOUND_ARGUMENT,
        TILE_UBOUND_ARGUMENT
    }

    # just use python to throw out all numeric values because I'm bad at
    # regular expressions.
    words = re.findall(r'\b(?:[\w]+)\b', lbound)
    # todo:: Probably not necessary
    words = [word for word in words if not word.isnumeric()]
    for word in words:
        if word not in keywords:
            raise NotImplementedError(
                f"{lbound} contained word not in {keywords}"
            )

    # find everything between a single set of parens to find math symbols.
    regexr = r'\(([^\)]+)\)'
    matches = re.findall(regexr, lbound)
    math_sym_string = lbound
    for match in matches:
        math_sym_string.replace(match, "")
    symbols = re.findall(r'[\+\-\/\*]', math_sym_string)

    # Replace each potential bound keyword inside of the string with its parts
    # This works because each bound keyword is guaranteed to be an IntVect.
    for idx, match in enumerate(matches):
        for keyword in keywords:
            if keyword in match:
                if (("IFELSE_K2D(" in match) or
                    ("IFELSE_K3D(" in match)):
                    matches[idx] = match.replace(
                        keyword, f'{keyword}.I();{keyword}.J();{keyword}.K()'
                    )
                else:
                    matches[idx] = match.replace(
                        keyword, f'{keyword}.I();IFELSE_K2D({keyword}.J(),1);IFELSE_K3D({keyword}.K(),1)'
                    )

    iterables = [match.split(';') for match in matches]
    if not iterables:
        raise RuntimeError(f"Nothing in lbound {lbound}.")

    # check if all lists inside the bounds equations are the same length.
    # Don't attempt to stitch different length arrays together.
    size = len(iterables[0])
    if not all([len(item) == size for item in iterables]):
        raise RuntimeError(f"Different lbound part sizes. {lbound}")

    # combine all lbound parts into 1.
    # list of mathematic expressions will always be 1 less than the number
    # of operands
    combined_bound = []
    for idx, values in enumerate(list(zip(*iterables))):
        combined_bound.append(values[0])
        if symbols:
            for i in range(1, len(values)):
                symbol = symbols[i-1]
                combined_bound[idx] += f'{symbol}{values[i]}'

    # remove whitespace
    combined_bound = [item.strip() for item in combined_bound]
    return combined_bound, words


def parse_extents(extents: str, src=None) -> list:
    """
    Parses an extents string.

    This assumes extents strings are of the format (x, y, z, ...).
    A list of integers separated by commas and surrounded by parenthesis.

    todo::
        * Source specific parsing should exist.

    :param str extents: The extents string to parse.
    :param str src: The optional source argument. If a grid source is given,
                    the function assumes extents to be a specific format.
    """
    if src:
        raise NotImplementedError("Source specific extents not implemented.")

    # default for parsing extents. Extents is assume to be a string
    # containing a list of integers surrounded by parentheses.
    if extents.count('(') != 1 or extents.count(')') != 1:
        raise IncorrectFormatException(
            f"Incorrect parenthesis placement for {extents}"
        )

    if not extents.startswith('(') or not extents.endswith(')'):
        raise IncorrectFormatException(
            f"{extents} is not the correct format of (x, y, z, ...)"
        )
    extents = extents.replace('(', '').replace(')', '')

    # isnumeric does not account for negative numbers.
    extents_list = [item.strip() for item in extents.split(',') if item]
    if any([(not item.lstrip('-').isnumeric()) for item in extents_list]):
        raise NonIntegerException(
            f"A value in the extents ({extents_list}) was not an integer."
        )

    # don't allow negative values for array sizes.
    if any([(int(item) < 0) for item in extents_list]):
        raise RuntimeError(
            f"A value in {extents_list} was negative."
        )

    return extents_list


def get_initial_index(vars_in: list, vars_out: list) -> int:
    """
    Returns the initial index based on a given variable masking.
    :param list vars_in: The variable masking for copying into the packet.
    :param list vars_out: The variable masking for copying out.
    :return: The size of the array given the variable masking.
    :rtype: int
    """
    starting = maxsize
    if not vars_in and not vars_out:
        raise TypeError("No variable masking for array in tf_spec.")

    starting_in = None
    if vars_in:
        starting_in = min(vars_in)
        starting = starting_in

    starting_out = None
    if vars_out:
        starting_out = min(vars_out)
        starting = starting_out

    if starting_in and starting_out:
        assert starting_in
        assert starting_out
        starting = min(starting_in, starting_out)

    if starting == maxsize:
        raise LogicError("Starting value for array is too large.")

    return starting


def get_array_size(vars_in: list, vars_out: list, check_out_size=False) -> int:
    """
    Returns the largest array size given a variable mask for copying in
    and copying out.

    :param list vars_in: The variable masking for copying into the packet.
    :param list vars_out: The variable masking for copying out.
    :param bool check_out_size: Flag to check if out array size is larger
                                than the in array size.
    :return: The size of the array given the variable masking.
    :rtype: int
    """
    largest = -maxsize
    if not vars_in and not vars_out:
        raise TypeError("No variable masking for given array in tf spec.")

    largest_in = None
    if vars_in:
        largest_in = max(vars_in)
        largest = largest_in

    largest_out = None
    if vars_out:
        largest_out = max(vars_out)
        largest = largest_out

    if vars_in and vars_out:
        assert largest_in is not None
        assert largest_out is not None

        # No test cases for a mariable mask in an out array that's
        # larger than the in mask. Need to create test cases or have
        # an existing use case.
        if check_out_size and largest_out > largest_in:
            raise NotImplementedError(
                "No test cases when vars_out is larger than vars_in!"
            )

        largest = max([largest_in, largest_out])

    if largest == -maxsize:
        raise LogicError("Negative array size, check variable masking.")
    return largest
