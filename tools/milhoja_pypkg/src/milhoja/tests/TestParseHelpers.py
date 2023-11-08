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
            self.assertFalse(True)

        with self.assertRaises(
            NonIntegerException,
            msg="parse_extents should have raise a NonIntegerException."
        ):
            input = "(1, 2, 3A, VBNVN)"
            parsed = parse_extents(input)
            self.assertFalse(True)

        # ...technically valid since we remove all parenthesis....
        # should this even be a valid input?
        with self.assertRaises(
            IncorrectFormatException,
            msg="Extents format should not be valid."
        ):
            input = "(((((((((((((((((()()()()1, 2, 3, 4,())))))))))))))))))"
            parsed = parse_extents(input)
            self.assertFalse(True)

        incorrect = ['']
        parsed = parse_extents("()")
        self.assertTrue(
            incorrect != parsed,
            f'"()" should return an empty list. Instead, {parsed}'
        )

        correct = []
        parsed = parse_extents("()")
        self.assertTrue(
            correct == parsed,
            f'"()" did not retrurn [], but {parsed}.'
        )

        correct = ['1', '1', '1', '1']
        parsed = parse_extents("(1,1,1,1)")
        self.assertTrue(
            correct == parsed, f"{parsed} did not return {correct}"
        )

        correct = ['1']
        parsed = parse_extents("(1)")
        self.assertTrue(
            correct == parsed, f'{parsed} did not return {correct}'
        )

    def check_bound(self, inp, generated, correct):
        self.assertTrue(
            generated == correct,
            f"{inp} returned {generated}, istead of {correct}."
        )

    def testLboundParser(self):
        inp = "(tile_lo, 5) - (1,1,1,1)"
        bound_array = parse_lbound(inp)
        correct = ["(lo)-IntVect{LIST_NDIM(1,1,1)}", "5-1"]
        self.check_bound(inp, bound_array, correct)

        lbound = "(tile_lo) + (1,1,1)"
        result = parse_lbound(lbound)
        correct = ["(lo)+IntVect{LIST_NDIM(1,1,1)}"]
        self.check_bound(lbound, result, correct)

        lbound = "(tile_lbound, 1) - (1,0,0,0)"
        result = parse_lbound(lbound)
        correct = ["(lbound)-IntVect{LIST_NDIM(1,0,0)}", "1-0"]
        self.check_bound(lbound, result, correct)

        lbound = "(tile_lo / 2)"
        result = parse_lbound(lbound)
        correct = ["(lo/2)"]
        self.check_bound(lbound, result, correct)

        lbound = "(1,1,1,1)"
        result = parse_lbound(lbound)
        correct = ['IntVect{LIST_NDIM(1,1,1)}', '1']
        self.check_bound(lbound, result, correct)

        with self.assertRaises(
            IncorrectFormatException,
            msg="Lbound has symbols that did not get caught."
        ):
            lbound = "(tile_lo) - (2,2,2) / 2"
            result = parse_lbound(lbound)
            self.assertFalse(True)
