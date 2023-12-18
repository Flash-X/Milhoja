import milhoja.tests
from milhoja.TemplateUtility import TemplateUtility


class TestTemplateUtility(milhoja.tests.TestCodeGenerators):
    """
    Unit test for utility functions in TemplateUtility class.
    """
    def testStartingIndex(self):
        # test none on both
        mask_in = None
        mask_out = None
        with self.assertRaises(TypeError):
            TemplateUtility.get_initial_index([], [])

        mask_in = [1, 2]
        mask_out = [1, 2]
        init = TemplateUtility.get_initial_index(mask_in, mask_out)
        self.assertTrue(init == 1)

        mask_in = [2, 8]
        mask_out = [2, 2]
        init = TemplateUtility.get_initial_index(mask_in, mask_out)
        self.assertTrue(init == 2)

        mask_in = [1, 10]
        mask_out = []
        init = TemplateUtility.get_initial_index(mask_in, mask_out)
        self.assertTrue(init == 1)

        mask_in = []
        mask_out = [3, 10]
        init = TemplateUtility.get_initial_index(mask_in, mask_out)
        self.assertTrue(init == 3)

    def testArraySize(self):
        # test none on both
        mask_in = None
        mask_out = None
        with self.assertRaises(TypeError):
            TemplateUtility.get_array_size([], [])

        mask_in = [1, 2]
        mask_out = [1, 2]
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 2)

        mask_in = [2, 8]
        mask_out = [2, 2]
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 8)

        with self.assertRaises(
            NotImplementedError,
            msg="No test cases for out size > in size, but no error thrown."
        ):
            mask_in = [1, 2]
            mask_out = [1, 6]
            TemplateUtility.get_array_size(mask_in, mask_out)

        mask_in = [1, 10]
        mask_out = []
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 10)

        mask_in = []
        mask_out = [1, 10]
        size = TemplateUtility.get_array_size(mask_in, mask_out)
        self.assertTrue(size == 10)
