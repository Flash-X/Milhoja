import re

from . import (
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT
)


class IncorrectFormatException(BaseException):
    pass


class NonIntegerException(BaseException):
    pass


def parse_lbound(lbound: str) -> list:
    """
    Parses an lbound string for use within the generator.
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


def parse_extents(extents: str) -> list:
    """
    Parses an extents string.

    This assumes extents strings are of the format (x, y, z, ...).
    A list of integers separated by commas and surrounded by parenthesis.
    """
    if extents.count('(') != 1 or extents.count(')') != 1:
        raise IncorrectFormatException(
            f"Incorrect parenthesis placement for {extents}"
        )

    if extents[0] != '(' or extents[-1] != ')':
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
