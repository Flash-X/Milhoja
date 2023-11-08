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

    def testLboundParser(self):
        ...