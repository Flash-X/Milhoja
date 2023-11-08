import re


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

    :param str lbound: The lbound string to parse.
    :param str data_source: The source of the data.
                            Eg: scratch or grid data.
    """
    # remove tile_ prefix from keywords
    lbound = lbound.replace("tile_", '').replace(' ', '')
    lbound_parts = []
    regexr = r'\(([^\)]+)\)'
    matches = re.findall(regexr, lbound)
    stitch = ''

    # find stitching arithmetic
    # ..todo::
    #    * allow math expressions inside intvect constructors?
    if len(matches) > 1:
        assert len(matches) == 2  # only allow simple math for now.
        symbols = re.findall(r'[\+\-\/\*]', lbound)
        if len(symbols) > 1:
            raise IncorrectFormatException(
                "lbound only supports one math op symbol for now."
            )
        stitch = symbols[0]

    for m in matches:
        m = m.split(',')
        assert len(m) > 0
        # we have a keyword
        if not m[0].isnumeric():
            ncomp = m[1] if len(m) > 1 else None
            lbound_parts.append((f'({m[0]})', ncomp))
        # easy case, all values are integers
        elif all([value.isnumeric() for value in m]):
            # We use this vector in case the scratch array dim < 3D.
            # All FArrayND constructors take an intvect as the first param.
            init_vect = ['1', '1', '1']
            ncomp = None
            for idx, value in enumerate(init_vect):
                init_vect[idx] = str(m[idx])
            if len(m) > len(init_vect):
                # there should never be a case where its greater than 4, since
                # FArray > 4D does not exist anyway.
                assert len(m) == 4
                ncomp = m[-1]
            # join together wrapped in an int vect
            lbound_parts.append(
                (f'IntVect{{LIST_NDIM({",".join(init_vect)})}}', ncomp)
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

    extents_list = [item.strip() for item in extents.split(',') if item]
    if any([(not item.isnumeric()) for item in extents_list]):
        raise NonIntegerException(
            f"A value in the extents ({extents_list}) was not an integer."
        )
    return extents_list
