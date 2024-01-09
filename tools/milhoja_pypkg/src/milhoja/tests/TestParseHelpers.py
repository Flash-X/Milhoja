import milhoja.tests

from milhoja.parse_helpers import parse_extents
from milhoja.parse_helpers import parse_lbound
from milhoja.parse_helpers import IncorrectFormatException
from milhoja.parse_helpers import NonIntegerException


class TestParseHelpers(milhoja.tests.TestCodeGenerators):
    """
    Unit test of extents parser function.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testExtentsParser(self):
        with self.assertRaises(
            IncorrectFormatException,
            msg="parse_extents did not raise an error."
        ):
            input = "1,2,3"
            parsed = parse_extents(input)

            input = "(1,2,3"
            parsed = parse_extents(input)

            input = "1,2,3)"
            parsed = parse_extents(input)

        with self.assertRaises(
            NonIntegerException,
            msg="parse_extents should have raise a NonIntegerException."
        ):
            input = "(1, 2, 3A, VBNVN)"
            parsed = parse_extents(input)

        with self.assertRaises(
            RuntimeError,
            msg="parse_extents should raise a RuntimeError if value is < 0."
        ):
            input = "(1, -2, 3, -4)"
            parsed = parse_extents(input)

        with self.assertRaises(
            IncorrectFormatException,
            msg="Extents format should not be valid."
        ):
            input = "(((((((((((((((((()()()()1, 2, 3, 4,())))))))))))))))))"
            parsed = parse_extents(input)

        incorrect = []
        parsed = parse_extents("()")
        self.assertTrue(
            incorrect == parsed,
            f'"()" should return an empty list. Instead, {parsed}'
        )

        inpt = "()"
        correct = []
        parsed = parse_extents(inpt)
        self.check_extents(inpt, parsed, correct)

        inpt = "(1,2,3,4)"
        correct = ['1', '2', '3', '4']
        parsed = parse_extents(inpt)
        self.check_extents(inpt, parsed, correct)

        inpt = "(1)"
        correct = ['1']
        parsed = parse_extents(inpt)
        self.check_extents(inpt, parsed, correct)

    def check_extents(self, input, generated, correct):
        """
        Alias for check_bound function. Reused for checking extents.

        :param list input: The input that created *generated*.
        :param list generated: The result after parsing input.
        :param list correct: The expected result.
        """
        self.check_bound(input, generated, correct)

    def check_bound(self, input, generated, correct):
        """
        A function that compares a generated to a correct input.

        :param list input: The input that created *generated*.
        :param list generated: The result after parsing input.
        :param list correct: The expected result.
        """
        self.assertTrue(
            generated == correct,
            f"{input} returned {generated}, istead of {correct}."
        )

    def testLboundParser(self):
        inp = "(tile_lo, 5) - (1,2,3,-4)"
        bound_array = parse_lbound(inp)
        correct = ["(lo)-IntVect{LIST_NDIM(1,2,3)}", "5--4"]
        self.check_bound(inp, bound_array, correct)

        with self.assertRaises(
            NotImplementedError,
            msg="This lbound type has not been implemented yet!"
        ):
            inp = "(5, tile_lo) - (1, 3, 4, 5)"
            parse_lbound(inp)

        with self.assertRaises(
            NotImplementedError,
            msg="This lbound type has not been implemented yet!"
        ):
            inp = "(tile_lo, 2) * (2, tile_lo)"
            parse_lbound(inp)

        with self.assertRaises(
            NotImplementedError,
            msg="This lbound type has not been implemented yet!"
        ):
            inp = "(3, 2, infinity, 4)"
            parse_lbound(inp)

        with self.assertRaises(
            NotImplementedError,
            msg="This lbound type has not been implemented yet!"
        ):
            inp = "(7, 7, infinity) + (tile_lo)"
            parse_lbound(inp)

        with self.assertRaises(
            NotImplementedError,
            msg="This lbound type has not been implemented yet!"
        ):
            inp = "(tile_lo, infinity) + (7, 7, 8, 2)"
            result = parse_lbound(inp)
            print(result)

        with self.assertRaises(
            NotImplementedError,
            msg="This lbound type has not been implemented yet!"
        ):
            inp = "(1, tile_lo, 4)"
            parse_lbound(inp)

        with self.assertRaises(
            NotImplementedError,
            msg="This lbound type has not been implemented yet!"
        ):
            inp = "(tile_lo, tile_hi)"
            parse_lbound(inp)

        inp = "(2*4+2-1, 32+2/2/3, 4, 1)"
        result = parse_lbound(inp)
        correct = ['IntVect{LIST_NDIM(2*4+2-1,32+2/2/3,4)}', '1']
        self.check_bound(inp, result, correct)

        inp = "(tile_lo, 3*7) + (2, 3, 4, 5)"
        result = parse_lbound(inp)
        correct = ["(lo)+IntVect{LIST_NDIM(2,3,4)}", "3*7+5"]
        self.check_bound(inp, result, correct)

        inp = "(7, 7, 7) * (tile_lo)"
        result = parse_lbound(inp)
        correct = ["IntVect{LIST_NDIM(7,7,7)}*(lo)"]
        self.check_bound(inp, result, correct)

        lbound = "(tile_lo) + (1,2,3)"
        result = parse_lbound(lbound)
        correct = ["(lo)+IntVect{LIST_NDIM(1,2,3)}"]
        self.check_bound(lbound, result, correct)

        lbound = "(tile_lbound, 3) - (1,0,-1,2)"
        result = parse_lbound(lbound)
        correct = ["(lbound)-IntVect{LIST_NDIM(1,0,-1)}", "3-2"]
        self.check_bound(lbound, result, correct)

        lbound = "(tile_lo / 2)"
        result = parse_lbound(lbound)
        correct = ["(lo/2)"]
        self.check_bound(lbound, result, correct)

        lbound = "(1,-2,-3,4)"
        result = parse_lbound(lbound)
        correct = ['IntVect{LIST_NDIM(1,-2,-3)}', '4']
        self.check_bound(lbound, result, correct)

        # this type of input would require a more complex math parser.
        with self.assertRaises(
            IncorrectFormatException,
            msg="Lbound has symbols that did not get caught."
        ):
            lbound = "(tile_lo) - (2,2,2) / 2"
            result = parse_lbound(lbound)