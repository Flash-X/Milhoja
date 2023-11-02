#!/usr/bin/env python
import re

from milhoja import TaskFunction

def test_parse_lbound():
    bound_array = parse_lbound("(tile_lo, 5) - (1,1,1,1)")
    assert bound_array == [ "(lo)-IntVect{LIST_NDIM(1,1,1)}", "5-1"]
    print("Test 1... Done")

    lbound = "(tile_lo) + (1,1,1)"
    result = parse_lbound(lbound)
    assert result == ["(lo)+IntVect{LIST_NDIM(1,1,1)}"]
    print("Test 2... Done")

    lbound = "(tile_lbound, 1) - (1,0,0,0)"
    result = parse_lbound(lbound)
    assert result == ["(lbound)-IntVect{LIST_NDIM(1,0,0)}", "1-0"]
    print("Test 3... Done")

    lbound = "(tile_lo / 2)"
    result = parse_lbound(lbound)
    assert result == ["(lo/2)"]

    lbound = "(1,1,1,1)"
    result = parse_lbound(lbound)
    assert result == ['IntVect{LIST_NDIM(1,1,1)}', '1']


def parse_lbound(lbound: str) -> list:
    """
    Parses an lbound string for use within the generator.
    ..todo::
        * This lbound parser only allows simple lbounds. The current format does not allow
          nested arithmetic expressions or more than 2 intvects being combined.
    
    :param str lbound: The lbound string to parse.
    :param str data_source: The source of the data. Eg: scratch or grid data. 
    """
    lbound = lbound.replace("tile_", '').replace(' ', '') # remove tile_ prefix from keywords
    lbound_parts = []
    regexr = r'\(([^\)]+)\)'
    matches = re.findall(regexr, lbound)
    stitch = ''
    
    # find stitching arithmetic
    # ..todo::
    #    * allow math expressions inside intvect constructors?
    #    * use dummy packet generator to test lbound parsing
    if len(matches) > 1:
        assert len(matches) == 2 # only allow simple math for now.
        symbols = re.findall(r'[\+\-\/\*]', lbound)
        assert len(symbols) == 1 # for now
        stitch = symbols[0]

    for m in matches:
        m = m.split(',')
        assert len(m) > 0
        if not m[0].isnumeric(): # we have a keyword
            ncomp = m[1] if len(m) > 1 else None
            lbound_parts.append((f'({m[0]})', ncomp))
        elif all([ value.isnumeric() for value in m ]):
            init_vect = ['1','1','1']
            ncomp = None
            for idx,value in enumerate(init_vect):
                init_vect[idx] = str(m[idx])
            if len(m) > len(init_vect):
                assert len(m) == 4 # there should never be a case where its greater than 4
                ncomp = m[-1]
            lbound_parts.append((f'IntVect{{LIST_NDIM({",".join(init_vect)})}}', ncomp))

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


if __name__ == "__main__":
    test_parse_lbound()